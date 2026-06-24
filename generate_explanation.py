import os
import json
import random
import time
from typing import Union, List, Dict, Optional
import subprocess
import argparse
import glob
from PIL import Image
import re
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None
import asyncio
import uuid # Import uuid for generating trace_id

from mllm_tools.litellm import LiteLLMWrapper
from mllm_tools.utils import _prepare_text_inputs # Keep _prepare_text_inputs if still used directly in main

# Import new modules
from src.core.explanation_planner import ExplanationPlanner
from src.core.code_generator import CodeGenerator
from src.core.explanation_renderer import ExplanationRenderer
from src.utils.utils import _print_response, _extract_code, extract_xml, parse_scene_outline_tokens # Import utility functions
from src.config.config import Config # Import Config class

# Explanation parsing
from src.core.parse_explanation import (
    get_images_from_explanation,
    image_with_most_non_black_space
)
from task_generator import get_banned_reasonings
from task_generator.prompts_raw import (_code_font_size, _code_disable, _code_limit, _prompt_manim_cheatsheet)

# Load allowed models list from JSON file
allowed_models_path = os.path.join(os.path.dirname(__file__), 'src', 'utils', 'allowed_models.json')
with open(allowed_models_path, 'r') as f:
    allowed_models = json.load(f).get("allowed_models", [])

if load_dotenv:
    # DOTENV_PATH lets a second model run (e.g. glm-5.2 via the gateway) point at
    # its own .env without disturbing the main checkout's minimax .env. Unset =>
    # None => default discovery, identical to before.
    load_dotenv(os.environ.get("DOTENV_PATH") or None, override=True)


def convert_explanation_to_gif(explanation_path: str, gif_path: str, max_duration: float = 3.0, fps: int = 10) -> bool:
    """Convert a explanation file to a short GIF (first few seconds only).

    Args:
        explanation_path: Path to input explanation
        gif_path: Path to output GIF
        max_duration: Maximum duration in seconds
        fps: Frames per second for GIF

    Returns:
        True if successful, False otherwise
    """
    try:
        import ffmpeg
        (
            ffmpeg
            .input(explanation_path, t=max_duration)
            .filter('fps', fps=fps)
            .filter('scale', 480, -1)
            .output(gif_path, loop=0)
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except Exception as e:
        print(f"Failed to convert to GIF: {e}")
        return False

class ExplanationGenerator:
    """
    A class for generating manim explanations using AI models.

    This class coordinates the explanation generation pipeline by managing scene planning,
    code generation, and explanation rendering. It supports concurrent scene processing,
    visual code fixing, and RAG (Retrieval Augmented Generation).

    Args:
        planner_model: Model used for scene planning and high-level decisions
        scene_model: Model used specifically for scene generation (defaults to planner_model)
        helper_model: Helper model for additional tasks (defaults to planner_model)
        output_dir (str): Directory to store generated files and explanations
        verbose (bool): Whether to print detailed output
        use_rag (bool): Whether to use Retrieval Augmented Generation
        use_context_learning (bool): Whether to use context learning with example code
        context_learning_path (str): Path to context learning examples
        chroma_db_path (str): Path to ChromaDB for RAG
        manim_docs_path (str): Path to Manim documentation for RAG
        embedding_model (str): Model to use for embeddings
        use_visual_fix_code (bool): Whether to use visual feedback for code fixing
        use_langfuse (bool): Whether to enable Langfuse logging
        trace_id (str, optional): Trace ID for logging
        max_scene_concurrency (int): Maximum number of scenes to process concurrently

    Attributes:
        output_dir (str): Directory for output files
        verbose (bool): Verbosity flag
        use_visual_fix_code (bool): Visual code fixing flag
        session_id (str): Unique session identifier
        scene_semaphore (asyncio.Semaphore): Controls concurrent scene processing
        banned_reasonings (list): List of banned reasoning patterns
        planner (ExplanationPlanner): Handles scene planning
        code_generator (CodeGenerator): Handles code generation
        explanation_renderer (ExplanationRenderer): Handles explanation rendering
    """

    def __init__(self,
                 planner_model,
                 scene_model=None,
                 helper_model=None,
                 output_dir="output",
                 verbose=False,
                 use_rag=False,
                 use_context_learning=False,
                 context_learning_path="data/context_learning",
                 chroma_db_path="data/rag/chroma_db",
                 manim_docs_path="data/rag/manim_docs",
                 embedding_model="azure/text-embedding-3-large",
                 use_visual_fix_code=False,
                 use_langfuse=True,
                 trace_id=None,
                 max_scene_concurrency: int = 5,
                 translate_to_chinese: bool = False):  # Default: False (no translation)
        self.output_dir = output_dir
        self.verbose = verbose
        self.use_visual_fix_code = use_visual_fix_code
        self.translate_to_chinese = translate_to_chinese  # Add translation flag
        self.session_id = self._load_or_create_session_id()  # Modified to load existing or create new
        self.scene_semaphore = asyncio.Semaphore(max_scene_concurrency)
        self.banned_reasonings = get_banned_reasonings()

        # Initialize separate modules
        self.planner = ExplanationPlanner(
            planner_model=planner_model,
            helper_model=helper_model,
            output_dir=output_dir,
            print_response=verbose,
            use_context_learning=use_context_learning,
            context_learning_path=context_learning_path,
            use_rag=use_rag,
            session_id=self.session_id,
            chroma_db_path=chroma_db_path,
            manim_docs_path=manim_docs_path,
            embedding_model=embedding_model,
            use_langfuse=use_langfuse
        )
        self.code_generator = CodeGenerator(
            scene_model=scene_model if scene_model is not None else planner_model,
            helper_model=helper_model if helper_model is not None else planner_model,
            output_dir=output_dir,
            print_response=verbose,
            use_rag=use_rag,
            use_context_learning=use_context_learning,
            context_learning_path=context_learning_path,
            chroma_db_path=chroma_db_path,
            manim_docs_path=manim_docs_path,
            embedding_model=embedding_model,
            use_visual_fix_code=use_visual_fix_code,
            use_langfuse=use_langfuse,
            session_id=self.session_id
        )
        self.explanation_renderer = ExplanationRenderer(
            output_dir=output_dir,
            print_response=verbose,
            use_visual_fix_code=use_visual_fix_code,
            scene_model=scene_model if scene_model is not None else planner_model
        )

        # Use the same model for translation as well
        self.translator_model = scene_model if scene_model is not None else planner_model

    def _load_or_create_session_id(self) -> str:
        """
        Load existing session ID from file or create a new one.

        Returns:
            str: The session ID either loaded from file or newly created.
        """
        session_file = os.path.join(self.output_dir, "session_id.txt")

        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_id = f.read().strip()
                print(f"Loaded existing session ID: {session_id}")
                return session_id

        # Create new session ID if none exists
        session_id = str(uuid.uuid4())
        os.makedirs(self.output_dir, exist_ok=True)
        with open(session_file, 'w') as f:
            f.write(session_id)
        print(f"Created new session ID: {session_id}")
        return session_id

    def _save_topic_session_id(self, topic: str, session_id: str) -> None:
        """
        Save session ID for a specific topic.

        Args:
            topic (str): The topic to save the session ID for
            session_id (str): The session ID to save
        """
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        topic_dir = os.path.join(self.output_dir, file_prefix)
        os.makedirs(topic_dir, exist_ok=True)

        session_file = os.path.join(topic_dir, "session_id.txt")
        with open(session_file, 'w') as f:
            f.write(session_id)

    def _load_topic_session_id(self, topic: str) -> Optional[str]:
        """
        Load session ID for a specific topic if it exists.

        Args:
            topic (str): The topic to load the session ID for

        Returns:
            Optional[str]: The session ID if found, None otherwise
        """
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        session_file = os.path.join(self.output_dir, file_prefix, "session_id.txt")

        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                return f.read().strip()
        return None


    async def generate_scene_implementation(self,
                                      topic: str,
                                      description: str,
                                      plan: str,
                                      session_id: str) -> List[str]:
        """
        Generate scene implementations using ExplanationPlanner.

        Args:
            topic (str): The topic of the explanation
            description (str): Description of the explanation content
            plan (str): The scene plan to implement
            session_id (str): Session identifier for tracking

        Returns:
            List[str]: List of generated scene implementations
        """
        return await self.planner.generate_scene_implementation(topic, description, plan, session_id)

    async def generate_scene_implementation_concurrently(self,
                                              topic: str,
                                              description: str,
                                              plan: str,
                                              session_id: str) -> List[str]:
        """
        Generate scene implementations concurrently using ExplanationPlanner.

        Args:
            topic (str): The topic of the explanation
            description (str): Description of the explanation content
            plan (str): The scene plan to implement
            session_id (str): Session identifier for tracking

        Returns:
            List[str]: List of generated scene implementations
        """
        return await self.planner.generate_scene_implementation_concurrently(topic, description, plan, session_id, self.scene_semaphore) # Pass semaphore

    def load_implementation_plans(self, topic: str) -> Dict[int, Optional[str]]:
        """
        Load implementation plans for each scene.

        Args:
            topic (str): The topic to load implementation plans for

        Returns:
            Dict[int, Optional[str]]: Dictionary mapping scene numbers to their plans.
                                    If a scene's plan is missing, its value will be None.
        """
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)

        # Load scene outline from file
        scene_outline_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        if not os.path.exists(scene_outline_path):
            return {}
        
        with open(scene_outline_path, "r") as f:
            scene_outline = f.read()

        # Extract scene outline to get number of scenes
        scene_outline_content = extract_xml(scene_outline)
        scene_number = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline_content))
        print(f"Number of scenes: {scene_number}")

        implementation_plans = {}

        # Check each scene's implementation plan
        for i in range(1, scene_number + 1):
            plan_path = os.path.join(self.output_dir, file_prefix, f"scene{i}", f"{file_prefix}_scene{i}_implementation_plan.txt")
            if os.path.exists(plan_path):
                with open(plan_path, "r") as f:
                    implementation_plans[i] = f.read()
                print(f"Found existing implementation plan for scene {i}")
            else:
                implementation_plans[i] = None
                print(f"Missing implementation plan for scene {i}")

        return implementation_plans

    async def render_explanation_fix_code(self,
                              topic: str,
                              description: str,
                              scene_outline: str,
                              implementation_plans: List,
                              max_retries=3,
                              session_id: str = None,
                              problem_image: Optional[Image.Image] = None) -> None:
        """
        Render the explanation for all scenes with code fixing capability.

        Args:
            topic (str): The topic of the explanation
            description (str): Description of the explanation content
            scene_outline (str): The overall scene outline
            implementation_plans (List): List of implementation plans for each scene
            max_retries (int, optional): Maximum number of code fix attempts. Defaults to 3.
            session_id (str, optional): Session identifier for tracking
            problem_image: Optional problem diagram image
        """
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)

        # Create tasks for each scene
        tasks = []
        for i, implementation_plan in enumerate(implementation_plans):
            # Try to load scene trace id, or generate new one if it doesn't exist
            scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{i+1}")
            subplan_dir = os.path.join(scene_dir, "subplans")
            os.makedirs(subplan_dir, exist_ok=True)  # Create directories if they don't exist
            
            scene_trace_id_path = os.path.join(subplan_dir, "scene_trace_id.txt")
            try:
                with open(scene_trace_id_path, 'r') as f:
                    scene_trace_id = f.read().strip()
            except FileNotFoundError:
                scene_trace_id = str(uuid.uuid4())
                with open(scene_trace_id_path, 'w') as f:
                    f.write(scene_trace_id)

            task = self.process_scene(i, scene_outline, implementation_plan, topic, description, max_retries, file_prefix, session_id, scene_trace_id, problem_image=problem_image)
            tasks.append(task)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

    async def process_scene(self, i: int, scene_outline: str, scene_implementation: str, topic: str, description: str, max_retries: int, file_prefix: str, session_id: str, scene_trace_id: str, problem_image: Optional[Image.Image] = None): # added scene_trace_id
        """
        Process a single scene using CodeGenerator and ExplanationRenderer.

        Args:
            i (int): Scene index
            scene_outline (str): Overall scene outline
            scene_implementation (str): Implementation plan for this scene
            topic (str): The topic of the explanation
            description (str): Description of the explanation content
            max_retries (int): Maximum number of code fix attempts
            file_prefix (str): Prefix for file naming
            session_id (str): Session identifier for tracking
            scene_trace_id (str): Trace identifier for this scene
            problem_image: Optional problem diagram image
        """
        # Record start time for this scene
        scene_start_time = time.perf_counter()

        curr_scene = i + 1
        rag_queries_cache = {}  # Initialize RAG queries cache

        # Create necessary directories
        code_dir = os.path.join(self.output_dir, file_prefix, f"scene{curr_scene}", "code")
        os.makedirs(code_dir, exist_ok=True)
        media_dir = os.path.join(self.output_dir, file_prefix, "media")
        scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{curr_scene}")

        # Check if already successfully rendered
        succ_rendered_path = os.path.join(scene_dir, "succ_rendered.txt")
        code_token_file = os.path.join(code_dir, f"scene{curr_scene}_code_tokens.json")
        if os.path.exists(succ_rendered_path):
            print(f"Scene {curr_scene} already successfully rendered, skipping")
            return

        # Initialize trace tokens bucket for this scene in scene_model (isolated per scene_trace_id)
        scene_model = self.code_generator.scene_model
        if scene_trace_id not in scene_model._trace_tokens:
            scene_model._trace_tokens[scene_trace_id] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Find highest existing version
        existing_versions = []
        if os.path.isdir(code_dir):
            for filename in os.listdir(code_dir):
                match = re.match(rf"{re.escape(file_prefix)}_scene{curr_scene}_v(\d+)\.py$", filename)
                if match:
                    existing_versions.append(int(match.group(1)))

        # Load existing code if available, otherwise prepare to generate new code
        if existing_versions:
            curr_version = max(existing_versions)
            print(f"Found existing code up to v{curr_version}, loading v{curr_version}")
            code_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py")
            with open(code_path, "r") as f:
                code = f.read()

            # If we already have versions beyond max_retries, try rendering once and stop
            if curr_version > max_retries:
                print(f"Scene {curr_scene} has {curr_version} versions (>{max_retries}), attempting final render")
                async with self.scene_semaphore:
                    code, curr_version, error_message = await self.explanation_renderer.render_scene(
                        code=code,
                        file_prefix=file_prefix,
                        curr_scene=curr_scene,
                        curr_version=curr_version,
                        code_dir=code_dir,
                        media_dir=media_dir,
                        max_visual_retries=1,
                        use_visual_fix_code=self.use_visual_fix_code,
                        visual_self_reflection_func=self.code_generator.visual_self_reflection,
                        banned_reasonings=self.banned_reasonings,
                        scene_trace_id=scene_trace_id,
                        topic=topic,
                        session_id=session_id,
                        implementation_plan=scene_implementation,
                        problem_image=problem_image
                    )
                    if error_message is not None:
                        print(f"Scene {curr_scene} final render failed: {error_message}")
                        with open(os.path.join(scene_dir, "render_failed.txt"), "w") as f:
                            f.write(f"Failed at v{curr_version}\nLast error:\n{error_message}")
                return
        else:
            # No existing code found, will need to generate
            curr_version = 0
            code = None

        async with self.scene_semaphore:
            # Only generate new code if we don't have any code yet
            if code is None:
                print(f"Generating new code for scene {curr_scene} v{curr_version}")
                code, log = await self.code_generator.generate_manim_code(
                    topic=topic,
                    description=description,
                    scene_outline=scene_outline,
                    scene_implementation=scene_implementation,
                    scene_number=curr_scene,
                    additional_context=[_prompt_manim_cheatsheet, _code_font_size, _code_limit, _code_disable],
                    scene_trace_id=scene_trace_id,
                    session_id=session_id,
                    problem_image=problem_image,
                    rag_queries_cache=rag_queries_cache,
                    file_prefix=file_prefix  # Pass file_prefix for Scene1 reference
                )

                with open(os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}_init_log.txt"), "w") as f:
                    f.write(log)
                initial_code_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py")
                with open(initial_code_path, "w") as f:
                    f.write(code)
                print(f"Code saved to {code_dir}/{file_prefix}_scene{curr_scene}_v{curr_version}.py")
            else:
                print(f"Using existing code for scene {curr_scene} v{curr_version}, will attempt to render")

            # Step 3B: Compile and fix code if needed
            error_message = None
            while True: # Retry loop controlled by break statements
                code, curr_version, error_message = await self.explanation_renderer.render_scene(
                    code=code,
                    file_prefix=file_prefix,
                    curr_scene=curr_scene,
                    curr_version=curr_version,
                    code_dir=code_dir,
                    media_dir=media_dir,
                    max_visual_retries=1,
                    use_visual_fix_code=self.use_visual_fix_code,
                    visual_self_reflection_func=self.code_generator.visual_self_reflection,
                    banned_reasonings=self.banned_reasonings,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    session_id=session_id,
                    implementation_plan=scene_implementation,
                    problem_image=problem_image
                )
                if error_message is None: # Render success if error_message is None
                    break

                if curr_version >= max_retries: # Max retries reached
                    print(f"Max retries reached for scene {curr_scene}, error: {error_message}")
                    # Mark scene as failed
                    with open(os.path.join(scene_dir, "render_failed.txt"), "w") as f:
                        f.write(f"Failed after {max_retries} retries\nLast error:\n{error_message}")
                    break # Exit retry loop

                curr_version += 1
                # if program runs this, it means that the code is not rendered successfully
                code, log = await self.code_generator.fix_code_errors(
                    implementation_plan=scene_implementation,
                    code=code,
                    error=error_message,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=curr_scene,
                    session_id=session_id,
                    rag_queries_cache=rag_queries_cache
                )

                with open(os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}_fix_log.txt"), "w") as f:
                    f.write(log)
                with open(os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py"), "w") as f:
                    f.write(code)

                print(f"Code saved to {code_dir}/{file_prefix}_scene{curr_scene}_v{curr_version}.py")

        # Calculate and log scene processing time
        scene_elapsed = time.perf_counter() - scene_start_time
        print(f"✓ Scene {curr_scene} completed in {scene_elapsed:.2f}s ({scene_elapsed/60:.2f} min)")

        # Save scene timing information
        timing_file = os.path.join(self.output_dir, file_prefix, "timing.json")

        # Load existing timing data
        timing_data = {}
        if os.path.exists(timing_file):
            try:
                with open(timing_file, 'r') as f:
                    timing_data = json.load(f)
            except:
                pass

        # Initialize scene_timings if not exists
        if "scene_timings" not in timing_data:
            timing_data["scene_timings"] = {}

        # Record this scene's timing
        timing_data["scene_timings"][f"scene_{curr_scene}"] = {
            "scene_number": curr_scene,
            "time_seconds": scene_elapsed,
            "time_minutes": scene_elapsed / 60,
            "final_version": curr_version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save updated timing data
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f, indent=2)

        # Record scene_model token usage after processing and save to file
        # Only write if file doesn't already exist (preserve data across reruns)
        if not os.path.exists(code_token_file):
            code_tokens = dict(scene_model._trace_tokens.get(scene_trace_id, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}))
            with open(code_token_file, 'w') as f:
                json.dump(code_tokens, f, indent=2)
            print(f"Scene {curr_scene} code token usage saved to {code_token_file}")


    def run_manim_process(self,
                          topic: str):
        """
        Run manim on all generated manim code for a specific topic using ExplanationRenderer.

        Args:
            topic (str): The topic to render explanations for
        """
        return self.explanation_renderer.run_manim_process(topic)

    def create_snapshot_scene(self, topic: str, scene_number: int, version_number: int, return_type: str = "image"):
        """
        Create a snapshot of the explanation for a specific topic and scene using ExplanationRenderer.

        Args:
            topic (str): The topic of the explanation
            scene_number (int): Scene number to snapshot
            version_number (int): Version number to snapshot
            return_type (str, optional): Type of snapshot to return. Defaults to "image".

        Returns:
            The snapshot in the specified format
        """
        return self.explanation_renderer.create_snapshot_scene(topic, scene_number, version_number, return_type)

    def combine_explanations(self, topic: str):
        """
        Combine all explanations and subtitle files for a specific topic using ExplanationRenderer.

        Args:
            topic (str): The topic to combine explanations for
        """
        self.explanation_renderer.combine_explanations(topic)

    async def _generate_scene_implementation_single(self, topic: str, description: str, scene_outline_i: str, i: int, file_prefix: str, session_id: str, scene_trace_id: str, problem_image: Optional[Image.Image] = None) -> str:
        """
        Generate detailed implementation plan for a single scene using ExplanationPlanner.

        Args:
            topic (str): The topic of the explanation
            description (str): Description of the explanation content
            scene_outline_i (str): Outline for this specific scene
            i (int): Scene index
            file_prefix (str): Prefix for file naming
            session_id (str): Session identifier for tracking
            scene_trace_id (str): Trace identifier for this scene
            problem_image: Optional problem diagram image

        Returns:
            str: Generated implementation plan
        """
        return await self.planner._generate_scene_implementation_single(topic, description, scene_outline_i, i, file_prefix, session_id, scene_trace_id, problem_image=problem_image)

    async def generate_markdown_diagrams(self, topic: str, description: str, max_retries: int, only_plan: bool = False, problem_image: Optional[Image.Image] = None):
        """Generate a Markdown document with last-frame PNG diagrams.

        Pipeline:
        1) Generate/load SCENE_OUTLINE (interleaved TEXT_k + SCENE_k)
        2) For each SCENE_k: generate implementation -> generate Manim code -> render last-frame PNG (manim -pql -s)
        3) Write solution.md interleaving TEXT blocks and embedded images.

        Args:
            topic: Topic name
            description: Problem description
            max_retries: Maximum code-fix retries
            only_plan: If True, only generate outline (and optionally implementation plans), skip rendering.
            problem_image: Optional problem diagram image to include in prompts
        """
        session_id = self._load_or_create_session_id()
        self._save_topic_session_id(topic, session_id)
        
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        
        # Load or generate scene outline
        scene_outline_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        outline_token_file = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline_tokens.json")

        if os.path.exists(scene_outline_path):
            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()
            print(f"Loaded existing scene outline for topic: {topic}")

            # Check if token file exists, if not create a placeholder indicating it was loaded
            if not os.path.exists(outline_token_file):
                placeholder_tokens = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "note": "Scene outline was generated in a previous run, tokens not tracked"
                }
                with open(outline_token_file, "w") as f:
                    json.dump(placeholder_tokens, f, indent=2)
                print(f"Created placeholder token file (scene outline from previous run)")

            if self.planner.use_rag:
                self.planner.relevant_plugins = self.planner.rag_integration.detect_relevant_plugins(topic, description) or []
                self.planner.rag_integration.set_relevant_plugins(self.planner.relevant_plugins)
                print(f"Detected relevant plugins: {self.planner.relevant_plugins}")
        else:
            print(f"Generating new scene outline for topic: {topic}")
            scene_outline = await self.planner.generate_scene_outline(topic, description, session_id, problem_image=problem_image)
            os.makedirs(os.path.join(self.output_dir, file_prefix), exist_ok=True)
            with open(scene_outline_path, "w") as f:
                f.write(scene_outline)
            # Token file is automatically created in generate_scene_outline()

        # Parse outline tokens (TEXT_k + SCENE_k in order)
        tokens = parse_scene_outline_tokens(scene_outline)
        scene_tokens = [t for t in tokens if t['type'] == 'scene']
        scene_numbers = sorted({t['k'] for t in scene_tokens})

        # Check if any scene has "No diagram" - if so, skip all rendering
        scene_outline_content = extract_xml(scene_outline)
        has_no_diagram_scene = False
        for scene_num in scene_numbers:
            scene_pattern = f'<SCENE_{scene_num}>(.*?)</SCENE_{scene_num}>'
            scene_match = re.search(scene_pattern, scene_outline_content, re.DOTALL | re.IGNORECASE)
            if scene_match:
                scene_content = scene_match.group(1)
                if 'no diagram' in scene_content.lower():
                    has_no_diagram_scene = True
                    print(f"✓ Detected 'No diagram' in Scene {scene_num} - will skip rendering and use original problem image")
                    break

        if has_no_diagram_scene and problem_image is not None:
            # Skip rendering, just save problem image as scene diagram
            print(f"⏩ Skipping all scene rendering, using original problem diagram")

            # Create doc directory and save problem image
            doc_dir = os.path.join(self.output_dir, file_prefix, "doc")
            os.makedirs(doc_dir, exist_ok=True)

            # Save problem image for the "No diagram" scene(s)
            for scene_num in scene_numbers:
                scene_pattern = f'<SCENE_{scene_num}>(.*?)</SCENE_{scene_num}>'
                scene_match = re.search(scene_pattern, scene_outline_content, re.DOTALL | re.IGNORECASE)
                if scene_match:
                    scene_content = scene_match.group(1)
                    if 'no diagram' in scene_content.lower():
                        doc_scene_path = os.path.join(doc_dir, f"scene{scene_num}.png")
                        problem_image.save(doc_scene_path)
                        print(f"✓ Saved problem diagram as scene{scene_num}.png")

            # Generate markdown without rendering
            await self._generate_markdown_without_rendering(
                topic=topic,
                scene_outline=scene_outline,
                file_prefix=file_prefix,
                session_id=session_id,
                problem_image=problem_image
            )
            return

        # Load or generate implementation plans (reference generate_explanation_pipeline behavior)
        implementation_plans_dict = self.load_implementation_plans(topic)
        if not implementation_plans_dict:
            # If none exist yet, initialize with all scenes from outline
            scene_outline_content = extract_xml(scene_outline)
            scene_count = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline_content))
            implementation_plans_dict = {i: None for i in range(1, scene_count + 1)}

        # In markdown mode, we only need plans for scenes that actually appear in outline tokens
        missing_scenes = []
        for scene_num in scene_numbers:
            if implementation_plans_dict.get(scene_num) is None:
                missing_scenes.append(scene_num)

        if missing_scenes:
            print(f"Generating implementation plans for missing scenes: {missing_scenes}")
            scene_outline_content = extract_xml(scene_outline)
            for scene_num in missing_scenes:
                # Only generate implementation plan for Scene 1
                # Scene 2+ will directly use Scene 1 code as reference
                if scene_num > 1:
                    print(f"Scene {scene_num}: Skipping implementation plan generation, will use Scene 1 code as reference")
                    implementation_plans_dict[scene_num] = f"<REFERENCE_TO_SCENE_1>Scene {scene_num} will be generated based on Scene 1 code for consistency</REFERENCE_TO_SCENE_1>"
                    # Create placeholder token file for Scene 2+
                    scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{scene_num}")
                    subplan_dir = os.path.join(scene_dir, "subplans")
                    os.makedirs(subplan_dir, exist_ok=True)
                    token_file = os.path.join(subplan_dir, f"scene{scene_num}_implementation_tokens.json")
                    if not os.path.exists(token_file):
                        placeholder_tokens = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "note": "Scene uses Scene 1 as reference, no separate implementation plan generated"
                        }
                        with open(token_file, "w") as f:
                            json.dump(placeholder_tokens, f, indent=2)
                    continue

                scene_match = re.search(f'<SCENE_{scene_num}>(.*?)</SCENE_{scene_num}>', scene_outline_content, re.DOTALL)
                if not scene_match:
                    continue
                scene_outline_i = scene_match.group(1)
                scene_trace_id = str(uuid.uuid4())
                implementation_plan = await self._generate_scene_implementation_single(
                    topic, description, scene_outline_i, scene_num, file_prefix, session_id, scene_trace_id, problem_image=problem_image
                )
                implementation_plans_dict[scene_num] = implementation_plan
        else:
            # All scenes have existing plans, create placeholder token files if they don't exist
            print(f"All implementation plans already exist, checking token files...")
            for scene_num in scene_numbers:
                scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{scene_num}")
                subplan_dir = os.path.join(scene_dir, "subplans")
                if os.path.exists(subplan_dir):
                    token_file = os.path.join(subplan_dir, f"scene{scene_num}_implementation_tokens.json")
                    if not os.path.exists(token_file):
                        placeholder_tokens = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "note": "Implementation plan was generated in a previous run, tokens not tracked"
                        }
                        with open(token_file, "w") as f:
                            json.dump(placeholder_tokens, f, indent=2)
                        print(f"  Created placeholder token file for scene {scene_num}")

        if only_plan:
            print(f"Only generating plans - skipping rendering for topic: {topic}")
            return

        # Render scenes (reuse generate_explanation_pipeline strategy: decide which scenes need processing)
        media_dir = os.path.join(self.output_dir, file_prefix, "media")
        doc_dir = os.path.join(self.output_dir, file_prefix, "doc")
        os.makedirs(doc_dir, exist_ok=True)

        rendered_paths: Dict[int, str] = {}

        # Convert dict to list maintaining scene order
        sorted_scene_numbers = sorted(implementation_plans_dict.keys())
        implementation_plans = [implementation_plans_dict[i] for i in sorted_scene_numbers]

        # In markdown mode, don't check succ_rendered.txt - process all scenes
        # This ensures we can re-export PNGs even if rendering was done previously
        scenes_to_process = []
        for i, implementation_plan in enumerate(implementation_plans):
            scene_num = i + 1
            if scene_num not in scene_numbers:
                continue
            scenes_to_process.append((scene_num, implementation_plan))

        if not scenes_to_process:
            print(f"No scenes to process for topic '{topic}'.")
        else:
            print(f"Processing {len(scenes_to_process)} scenes...")
            # Create a list of tuples with scene numbers and plans
            scene_plans = [(scene_num, plan) for scene_num, plan in scenes_to_process]
            # Sort by scene number to ensure correct order
            scene_plans.sort(key=lambda x: x[0])
            # Extract plans
            filtered_implementation_plans = [plan for _, plan in scene_plans]
            await self.render_explanation_fix_code(
                topic, description, scene_outline, filtered_implementation_plans,
                max_retries=max_retries, session_id=session_id, problem_image=problem_image
            )

        # Export PNGs for all scenes in outline (whether newly rendered or already existing)
        for scene_num in scene_numbers:
            # Find the latest rendered version by checking code files
            code_dir = os.path.join(self.output_dir, file_prefix, f"scene{scene_num}", "code")
            version_number = 0
            if os.path.isdir(code_dir):
                versions = []
                for fn in os.listdir(code_dir):
                    m = re.match(rf"{re.escape(file_prefix)}_scene{scene_num}_v(\d+)\.py$", fn)
                    if m:
                        versions.append(int(m.group(1)))
                if versions:
                    version_number = max(versions)

            exported = self.explanation_renderer.export_scene_png_to_doc(
                media_dir=media_dir,
                file_prefix=file_prefix,
                scene_number=scene_num,
                version_number=version_number,
            )
            if exported:
                rendered_paths[scene_num] = exported

        # Write Markdown interleaving TEXT and SCENE blocks
        md_path = os.path.join(doc_dir, "solution.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {topic}\n\n")

            # Generate markdown content (always in English first)
            for t in tokens:
                if t['type'] == 'text':
                    # Remove leading indentation that would turn text into Markdown code blocks
                    text = t['content'].strip("\n")
                    text = "\n".join(line.lstrip() for line in text.splitlines())
                    f.write(text.strip() + "\n\n")
                else:
                    scene_num = t['k']

                    # Check if scene outline says "No diagram" - use original problem image
                    scene_outline_content = extract_xml(scene_outline)
                    scene_pattern = f'<SCENE_{scene_num}>(.*?)</SCENE_{scene_num}>'
                    scene_match = re.search(scene_pattern, scene_outline_content, re.DOTALL | re.IGNORECASE)

                    use_original_diagram = False
                    if scene_match:
                        scene_content = scene_match.group(1)
                        if 'no diagram' in scene_content.lower():
                            use_original_diagram = True
                            print(f"Scene {scene_num}: Detected 'No diagram', will use original problem image")

                    # If "No diagram" detected, copy problem_diagram.png to scene
                    if use_original_diagram and problem_image is not None:
                        # Save problem image as this scene's diagram
                        doc_scene_path = os.path.join(doc_dir, f"scene{scene_num}.png")
                        problem_image.save(doc_scene_path)
                        print(f"✓ Copied problem diagram to {doc_scene_path}")
                        f.write(f'<p align="center">\n  <img src="scene{scene_num}.png" alt="Problem Diagram">\n</p>\n\n')
                        continue

                    # Check if scene failed
                    scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{scene_num}")
                    failed_marker = os.path.join(scene_dir, "render_failed.txt")

                    if os.path.exists(failed_marker):
                        f.write(f"**[Scene {scene_num} rendering failed - diagram unavailable]**\n\n")
                    else:
                        img_path = rendered_paths.get(scene_num)
                        if img_path:
                            f.write(f"![](scene{scene_num}.png)\n\n")
                        else:
                            f.write(f"**[Missing diagram for Scene {scene_num}]**\n\n")

        print(f"Markdown generated: {md_path}")

        # If translation is enabled, translate the complete markdown document
        if self.translate_to_chinese:
            await self._translate_markdown_document(md_path, session_id=session_id)

    async def _generate_markdown_without_rendering(self, topic: str, scene_outline: str, file_prefix: str, session_id: str, problem_image: Optional[Image.Image] = None):
        """Generate markdown using original problem image instead of rendering scenes.

        This is called when scene outline contains 'No diagram'.
        """
        doc_dir = os.path.join(self.output_dir, file_prefix, "doc")
        os.makedirs(doc_dir, exist_ok=True)

        # Parse outline tokens
        tokens = parse_scene_outline_tokens(scene_outline)

        # Extract scene outline content for checking
        scene_outline_content = extract_xml(scene_outline)

        # Write Markdown
        md_path = os.path.join(doc_dir, "solution_no_diagram.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {topic}\n\n")
            for t in tokens:
                if t['type'] == 'text':
                    # Remove leading indentation
                    text = t['content'].strip("\n")
                    text = "\n".join(line.lstrip() for line in text.splitlines())
                    f.write(text.strip() + "\n\n")
                else:
                    scene_num = t['k']
                    # Check if this scene has "No diagram"
                    scene_pattern = f'<SCENE_{scene_num}>(.*?)</SCENE_{scene_num}>'
                    scene_match = re.search(scene_pattern, scene_outline_content, re.DOTALL | re.IGNORECASE)

                    if scene_match:
                        scene_content = scene_match.group(1)
                        if 'no diagram' in scene_content.lower():
                            # Use original problem image (already saved)
                            f.write(f"![](scene{scene_num}.png)\n\n")
                        else:
                            # No diagram for this scene
                            f.write("\n")

        print(f"✓ Markdown generated (no rendering): {md_path}")

        # If translation is enabled, translate the complete markdown document
        if self.translate_to_chinese:
            await self._translate_markdown_document(md_path, session_id=session_id)

    async def _translate_markdown_document(self, md_path: str, session_id: str):
        """Translate the entire markdown document to Chinese.

        Args:
            md_path: Path to the original markdown file (solution.md)
            session_id: Session identifier for logging
        """
        try:
            # Read the markdown content
            with open(md_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            print(f"📝 Translating markdown document to Chinese...")

            prompt = (
                "You are a professional technical translator specializing in Markdown documents.\n\n"
                "**Task**: Translate the following complete Markdown document from English to Chinese.\n\n"
                "**Strict Rules**:\n"
                "1. **DO NOT modify**:\n"
                "   - LaTeX math expressions: `$...$`, `$$...$$`, `\\(...\\)`, `\\[...\\]`\n"
                "   - Code blocks: fenced (```) or indented code blocks\n"
                "   - Inline code: `...`\n"
                "   - Image references: `![...](...)` - keep the paths unchanged\n"
                "   - URLs and file paths\n"
                "   - HTML tags and comments\n"
                "   - Markdown syntax characters: `#`, `*`, `-`, `>`, `|`, etc.\n\n"
                "2. **Translate**:\n"
                "   - All natural language text from English to Chinese\n"
                "   - Headings, paragraphs, list items, table content\n"
                "   - Link text (but NOT URLs): `[translate this](keep-url-unchanged)`\n"
                "   - Image alt text (but NOT paths): `![translate alt](keep-path)`\n"
                "   - Keep technical terms accurate and consistent\n\n"
                "3. **Preserve Structure**:\n"
                "   - Keep all heading levels (`#`, `##`, etc.)\n"
                "   - Maintain list structure and indentation\n"
                "   - Preserve table formatting and alignment\n"
                "   - Keep all blank lines for paragraph separation\n"
                "   - Maintain bold `**text**` and italic `*text*` markers\n\n"
                "4. **Formatting**:\n"
                "   - Add spaces between Chinese characters and English/numbers when appropriate\n"
                "   - Use Chinese punctuation marks（，。；：！？）\n"
                "   - Keep line breaks that affect Markdown rendering\n\n"
                "5. **Output Requirements**:\n"
                "   - Return ONLY the translated Markdown content\n"
                "   - NO wrapping in code blocks or other formatting\n"
                "   - NO explanatory text, comments, or prefixes\n"
                "   - The output must be valid Markdown that renders correctly\n\n"
                "---\n\n"
                f"{original_content}\n\n"
                "---\n"
            )

            translated_content = await self.translator_model(
                _prepare_text_inputs(prompt),
                metadata={
                    "generation_name": "translate_full_markdown_to_zh",
                    "tags": ["translate", "zh", "markdown"],
                    "session_id": session_id,
                },
            )

            # Save Chinese version to solution_chinese.md (keep original solution.md)
            doc_dir = os.path.dirname(md_path)
            chinese_md_path = os.path.join(doc_dir, "solution_chinese.md")

            with open(chinese_md_path, 'w', encoding='utf-8') as f:
                f.write(translated_content.strip() + "\n")

            print(f"✓ Chinese translation saved to: {chinese_md_path}")
            print(f"  Original English version kept at: {md_path}")

        except Exception as e:
            print(f"⚠️  Translation failed: {e}")
            print(f"   Original English version available at: {md_path}")


    def check_theorem_status(self, theorem: Dict) -> Dict[str, bool]:
        """
        Check if a theorem has its plan, code files, and rendered explanations with detailed scene status.

        Args:
            theorem (Dict): Dictionary containing theorem information

        Returns:
            Dict[str, bool]: Dictionary containing status information for the theorem
        """
        topic = theorem['theorem']
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        
        # Check scene outline
        scene_outline_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        has_scene_outline = os.path.exists(scene_outline_path)
        
        # Get number of scenes if outline exists
        num_scenes = 0
        if has_scene_outline:
            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()
            scene_outline_content = extract_xml(scene_outline)
            num_scenes = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline_content))
        
        # Check implementation plans, code files, and rendered explanations
        implementation_plans = 0
        code_files = 0
        rendered_scenes = 0
        
        # Track status of individual scenes
        scene_status = []
        for i in range(1, num_scenes + 1):
            scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{i}")
            
            # Check implementation plan
            plan_path = os.path.join(scene_dir, f"{file_prefix}_scene{i}_implementation_plan.txt")
            has_plan = os.path.exists(plan_path)
            if has_plan:
                implementation_plans += 1
            
            # Check code files
            code_dir = os.path.join(scene_dir, "code")
            has_code = False
            if os.path.exists(code_dir):
                if any(f.endswith('.py') for f in os.listdir(code_dir)):
                    has_code = True
                    code_files += 1
            
            # Check rendered scene explanation
            has_render = False
            if os.path.exists(scene_dir):
                succ_rendered_path = os.path.join(scene_dir, "succ_rendered.txt")
                if os.path.exists(succ_rendered_path):
                    has_render = True
                    rendered_scenes += 1
            
            scene_status.append({
                'scene_number': i,
                'has_plan': has_plan,
                'has_code': has_code,
                'has_render': has_render
            })

        # Check combined explanation
        combined_explanation_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_combined.mp4")
        has_combined_explanation = os.path.exists(combined_explanation_path)
        
        return {
            'topic': topic,
            'has_scene_outline': has_scene_outline,
            'total_scenes': num_scenes,
            'implementation_plans': implementation_plans,
            'code_files': code_files,
            'rendered_scenes': rendered_scenes,
            'has_combined_explanation': has_combined_explanation,
            'scene_status': scene_status
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Manim explanations using AI')
    parser.add_argument('--model', type=str, choices=allowed_models,
                      default='gemini/gemini-1.5-pro-002', help='Select the AI model to use')
    parser.add_argument('--problem_path', type=str, required=True, help='Path to math problem JSON (list of problems)')
    parser.add_argument('--index', type=int, default=None, help='Index of specific problem to process (0-based). If not specified, process all problems.')
    parser.add_argument('--use_reference_solution', action='store_true', help='Append reference solution as hints (do not copy verbatim)')
    parser.add_argument('--helper_model', type=str, choices=allowed_models,
                      default=None, help='Select the helper model to use')
    # Explanation-specific modes removed: this script now generates Markdown + last-frame PNG diagrams only
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR, help='Output directory') # Use Config
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--max_retries', type=int, default=2, help='Maximum number of retries for code generation')
    parser.add_argument('--use_rag', '--rag', action='store_true', help='Use Retrieval Augmented Generation')
    parser.add_argument('--use_visual_fix_code','--visual_fix_code', action='store_true', help='Use VLM to fix code with rendered visuals')
    parser.add_argument('--chroma_db_path', type=str, default=Config.CHROMA_DB_PATH, help="Path to Chroma DB") # Use Config
    parser.add_argument('--manim_docs_path', type=str, default=Config.MANIM_DOCS_PATH, help="Path to manim docs") # Use Config
    parser.add_argument('--embedding_model', type=str,
                       default=Config.EMBEDDING_MODEL, # Use Config
                       choices=["azure/text-embedding-3-large", "vertex_ai/text-embedding-005"],
                       help='Select the embedding model to use')
    parser.add_argument('--use_context_learning', action='store_true',
                       help='Use context learning with example Manim code')
    parser.add_argument('--context_learning_path', type=str,
                       default=Config.CONTEXT_LEARNING_PATH, # Use Config
                       help='Path to context learning examples')
    parser.add_argument('--use_langfuse', action='store_true',
                       help='Enable Langfuse logging')
    parser.add_argument('--max_scene_concurrency', type=int, default=1, help='Maximum number of scenes to process concurrently')
    parser.add_argument('--max_topic_concurrency', type=int, default=1,
                       help='Maximum number of topics to process concurrently')
    parser.add_argument('--only_plan', action='store_true', help='Only generate scene outline and implementation plans')
    parser.add_argument('--translate_to_chinese', action='store_true',
                       help='Translate text content to Chinese in the final Markdown output (default: False)')
    args = parser.parse_args()

    # Initialize planner model using LiteLLM
    if args.verbose:
        verbose = True
    else:
        verbose = False
    planner_model = LiteLLMWrapper(
        model_name=args.model,
        temperature=0.7,
        print_cost=True,
        verbose=verbose,
        use_langfuse=args.use_langfuse
    )
    helper_model = LiteLLMWrapper(
        model_name=args.helper_model if args.helper_model else args.model, # Use helper_model if provided, else planner_model
        temperature=0.7,
        print_cost=True,
        verbose=verbose,
        use_langfuse=args.use_langfuse
    )
    scene_model = LiteLLMWrapper( # Initialize scene_model separately
        model_name=args.model,
        temperature=0.7,
        print_cost=True,
        verbose=verbose,
        use_langfuse=args.use_langfuse
    )
    print(f"Planner model: {args.model}, Helper model: {args.helper_model if args.helper_model else args.model}, Scene model: {args.model}") # Print all models


    # Load math problems (JSON list)
    with open(args.problem_path, "r") as f:
        problems = json.load(f)

    if not isinstance(problems, list) or not problems:
        raise ValueError("problem_path must point to a non-empty JSON list")

    # Filter problems based on --index argument
    if args.index is not None:
        if args.index < 0 or args.index >= len(problems):
            raise IndexError(f"--index {args.index} out of range for problems list of size {len(problems)}")
        problems_to_process = [(args.index, problems[args.index])]
        print(f"Processing single problem at index {args.index}")
    else:
        problems_to_process = list(enumerate(problems))
        print(f"Processing all {len(problems)} problems")

    explanation_generator = ExplanationGenerator(
        planner_model=planner_model,
        scene_model=scene_model,
        helper_model=helper_model,
        output_dir=args.output_dir,
        verbose=args.verbose,
        use_rag=args.use_rag,
        use_context_learning=args.use_context_learning,
        context_learning_path=args.context_learning_path,
        chroma_db_path=args.chroma_db_path,
        manim_docs_path=args.manim_docs_path,
        embedding_model=args.embedding_model,
        use_visual_fix_code=args.use_visual_fix_code,
        use_langfuse=args.use_langfuse,
        max_scene_concurrency=args.max_scene_concurrency,
        translate_to_chinese=args.translate_to_chinese
    )

    topic_semaphore = asyncio.Semaphore(args.max_topic_concurrency)

    def _build_description(prob: Dict) -> str:
        problem_text = prob.get("problem") or prob.get("question")
        if not problem_text:
            raise ValueError("Each problem item must contain a non-empty 'problem' or 'question' field")

        description_lines = [
            problem_text,
        ]

        if args.use_reference_solution:
            try:
                solution = prob["format_answer"]["format_solution"]
            except (KeyError, TypeError):
                solution = None
            if solution:
                description_lines += [
                    "",
                    "Reference solution (for correctness checking only; do NOT copy verbatim):",
                    solution,
                ]

        return "\n".join(description_lines)

    def _extract_problem_image(prob: Dict) -> Optional[Image.Image]:
        """Extract and decode base64 image from problem data if available."""
        img_data = prob.get("img")
        if not img_data:
            return None

        try:
            import base64
            from io import BytesIO
            # Decode base64 image
            img_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_bytes))
            return img
        except Exception as e:
            print(f"Warning: Failed to decode problem image: {e}")
            return None

    def _build_topic(prob: Dict, idx: int) -> str:
        problem_type = (prob.get("subject") or "math").strip().lower()
        type_slug = re.sub(r"[^a-z0-9_]+", "_", problem_type)
        return f"problem_{idx}_{type_slug}"

    async def _process_one_problem(prob: Dict, idx: int) -> None:
        topic = _build_topic(prob, idx)
        description = _build_description(prob)
        problem_image = _extract_problem_image(prob)
        if problem_image:
            description += "\n(Note: The attached image is the original diagram illustrating the problem setup.)"

        async with topic_semaphore:
            print(f"Processing problem index {idx}: {topic}")
            if problem_image:
                print(f"Problem has diagram image (size: {problem_image.size})")

                # Save the original problem image to output directory
                file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
                output_topic_dir = os.path.join(explanation_generator.output_dir, file_prefix)
                os.makedirs(output_topic_dir, exist_ok=True)

                problem_image_path = os.path.join(output_topic_dir, "problem_diagram.png")
                problem_image.save(problem_image_path)
                print(f"✓ Problem diagram saved to {problem_image_path}")

            # Record start time for this problem
            problem_start_time = time.perf_counter()

            try:
                await explanation_generator.generate_markdown_diagrams(
                    topic,
                    description,
                    max_retries=args.max_retries,
                    only_plan=args.only_plan,
                    problem_image=problem_image,
                )
            except Exception as e:
                print(f"✗ Problem {idx} ({topic}) failed and will be skipped: {e}")
                return

            # Calculate and log problem processing time
            problem_elapsed = time.perf_counter() - problem_start_time

            # Save timing information to file
            file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
            timing_file = os.path.join(explanation_generator.output_dir, file_prefix, "timing.json")
            os.makedirs(os.path.dirname(timing_file), exist_ok=True)

            # Load existing timing data to preserve scene_timings
            timing_data = {}
            if os.path.exists(timing_file):
                try:
                    with open(timing_file, 'r') as f:
                        timing_data = json.load(f)
                except Exception as e:
                    print(f"  Warning: Could not load existing timing data: {e}")

            # Load planner token files (scene outline + implementation plans per scene)
            planner_detailed_tokens = {"scene_outline": {}, "implementation_plans": {}}
            outline_token_file = os.path.join(explanation_generator.output_dir, file_prefix, f"{file_prefix}_scene_outline_tokens.json")
            if os.path.exists(outline_token_file):
                try:
                    with open(outline_token_file, 'r') as f:
                        planner_detailed_tokens["scene_outline"] = json.load(f)
                except Exception as e:
                    print(f"  Warning: Could not load scene outline tokens: {e}")

            # Load scene_model code token files and implementation plan tokens per scene
            scene_dirs = sorted([
                d for d in os.listdir(os.path.join(explanation_generator.output_dir, file_prefix))
                if os.path.isdir(os.path.join(explanation_generator.output_dir, file_prefix, d)) and d.startswith("scene")
            ])
            planner_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            scene_model_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            scene_model_detailed = {}

            for scene_dir in scene_dirs:
                scene_num = scene_dir.replace("scene", "")
                base = os.path.join(explanation_generator.output_dir, file_prefix, scene_dir)

                # Planner implementation tokens
                impl_token_file = os.path.join(base, "subplans", f"scene{scene_num}_implementation_tokens.json")
                if os.path.exists(impl_token_file):
                    try:
                        with open(impl_token_file, 'r') as f:
                            t = json.load(f)
                        planner_detailed_tokens["implementation_plans"][f"scene_{scene_num}"] = t
                        for k in ("input_tokens", "output_tokens", "total_tokens"):
                            planner_total[k] += t.get(k, 0)
                    except Exception as e:
                        print(f"  Warning: Could not load scene {scene_num} implementation tokens: {e}")

                # Scene model code tokens
                code_token_file = os.path.join(base, "code", f"scene{scene_num}_code_tokens.json")
                if os.path.exists(code_token_file):
                    try:
                        with open(code_token_file, 'r') as f:
                            t = json.load(f)
                        scene_model_detailed[f"scene_{scene_num}"] = t
                        for k in ("input_tokens", "output_tokens", "total_tokens"):
                            scene_model_total[k] += t.get(k, 0)
                    except Exception as e:
                        print(f"  Warning: Could not load scene {scene_num} code tokens: {e}")

            # Add scene outline tokens to planner total
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                planner_total[k] += planner_detailed_tokens["scene_outline"].get(k, 0)

            total_input_tokens = planner_total["input_tokens"] + scene_model_total["input_tokens"]
            total_output_tokens = planner_total["output_tokens"] + scene_model_total["output_tokens"]
            total_tokens = planner_total["total_tokens"] + scene_model_total["total_tokens"]

            print(f"✓ Problem {idx} ({topic}) completed in {problem_elapsed:.2f}s ({problem_elapsed/60:.2f} min)")
            print(f"  Token usage - Total: {total_tokens} (Input: {total_input_tokens}, Output: {total_output_tokens})")
            print(f"  Planner: {planner_total['total_tokens']}, Scene: {scene_model_total['total_tokens']}")

            timing_data.update({
                "problem_index": idx,
                "topic": topic,
                "total_time_seconds": problem_elapsed,
                "total_time_minutes": problem_elapsed / 60,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

            # Only update token_usage if not already present in existing timing data
            if "token_usage" not in timing_data:
                timing_data["token_usage"] = {
                    "total_tokens": total_tokens,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "planner_model": planner_total,
                    "planner_model_detailed": planner_detailed_tokens,
                    "scene_model": scene_model_total,
                    "scene_model_detailed": scene_model_detailed,
                }

            with open(timing_file, 'w') as f:
                json.dump(timing_data, f, indent=2)
            print(f"Timing data saved to {timing_file}")

    async def _run_all() -> None:
        start = time.perf_counter()

        tasks = [
            asyncio.create_task(_process_one_problem(prob, idx))
            for idx, prob in problems_to_process
        ]
        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start
        print(f"Total elapsed time: {elapsed:.2f}s")

    asyncio.run(_run_all())
