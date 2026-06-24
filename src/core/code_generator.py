import os
import re
import json
from typing import Union, List, Dict, Optional
from PIL import Image
import glob

from src.utils.utils import extract_json
from mllm_tools.utils import _prepare_text_inputs
from mllm_tools.gemini import GeminiWrapper
from mllm_tools.vertex_ai import VertexAIWrapper
from task_generator import (
    get_prompt_code_generation,
    get_prompt_fix_error,
    get_prompt_visual_fix_error,
    get_banned_reasonings,
    get_prompt_rag_query_generation_fix_error,
    get_prompt_context_learning_code,
    get_prompt_rag_query_generation_code
)
from task_generator.prompts_raw import (
    _code_font_size,
    _code_disable,
    _code_limit,
    _prompt_manim_cheatsheet
)
from src.rag.vector_store import RAGVectorStore # Import RAGVectorStore

class CodeGenerator:
    """A class for generating and managing Manim code."""

    def __init__(self, scene_model, helper_model, output_dir="output", print_response=False, use_rag=False, use_context_learning=False, context_learning_path="data/context_learning", chroma_db_path="rag/chroma_db", manim_docs_path="rag/manim_docs", embedding_model="azure/text-embedding-3-large", use_visual_fix_code=False, use_langfuse=True, session_id=None):
        """Initialize the CodeGenerator.

        Args:
            scene_model: The model used for scene generation
            helper_model: The model used for helper tasks
            output_dir (str, optional): Directory for output files. Defaults to "output".
            print_response (bool, optional): Whether to print model responses. Defaults to False.
            use_rag (bool, optional): Whether to use RAG. Defaults to False.
            use_context_learning (bool, optional): Whether to use context learning. Defaults to False.
            context_learning_path (str, optional): Path to context learning examples. Defaults to "data/context_learning".
            chroma_db_path (str, optional): Path to ChromaDB. Defaults to "rag/chroma_db".
            manim_docs_path (str, optional): Path to Manim docs. Defaults to "rag/manim_docs".
            embedding_model (str, optional): Name of embedding model. Defaults to "azure/text-embedding-3-large".
            use_visual_fix_code (bool, optional): Whether to use visual code fixing. Defaults to False.
            use_langfuse (bool, optional): Whether to use Langfuse logging. Defaults to True.
            session_id (str, optional): Session identifier. Defaults to None.
        """
        self.scene_model = scene_model
        self.helper_model = helper_model
        self.output_dir = output_dir
        self.print_response = print_response
        self.use_rag = use_rag
        self.use_context_learning = use_context_learning
        self.context_learning_path = context_learning_path
        self.context_examples = self._load_context_examples() if use_context_learning else None
        self.manim_docs_path = manim_docs_path

        self.use_visual_fix_code = use_visual_fix_code
        self.banned_reasonings = get_banned_reasonings()
        self.session_id = session_id # Use session_id passed from ExplanationGenerator

        if use_rag:
            self.vector_store = RAGVectorStore(
                chroma_db_path=chroma_db_path,
                manim_docs_path=manim_docs_path,
                embedding_model=embedding_model,
                session_id=self.session_id,
                use_langfuse=use_langfuse
            )
        else:
            self.vector_store = None

    def _load_context_examples(self) -> str:
        """Load all context learning examples from the specified directory.

        Returns:
            str: Formatted context learning examples, or None if no examples found.
        """
        examples = []
        for example_file in glob.glob(f"{self.context_learning_path}/**/*.py", recursive=True):
            with open(example_file, 'r') as f:
                examples.append(f"# Example from {os.path.basename(example_file)}\n{f.read()}\n")

        # Format examples using get_prompt_context_learning_code instead of _prompt_context_learning
        if examples:
            formatted_examples = get_prompt_context_learning_code(
                examples="\n".join(examples)
            )
            return formatted_examples
        return None

    async def _generate_rag_queries_code(self, implementation: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries from the implementation plan.

        Args:
            implementation (str): The implementation plan text
            scene_trace_id (str, optional): Trace ID for the scene. Defaults to None.
            topic (str, optional): Topic of the scene. Defaults to None.
            scene_number (int, optional): Scene number. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list.

        Returns:
            List[str]: List of generated RAG queries
        """
        # Create a cache key for this scene
        cache_key = f"{topic}_scene{scene_number}"

        # Check if we already have a cache file for this scene
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_code.json")

        # If cache file exists, load and return cached queries
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_queries = json.load(f)
                print(f"Using cached RAG queries for {cache_key}")
                return cached_queries

        # Generate new queries if not cached
        if relevant_plugins:
            prompt = get_prompt_rag_query_generation_code(implementation, ", ".join(relevant_plugins))
        else:
            prompt = get_prompt_rag_query_generation_code(implementation, "No plugins are relevant.")

        queries = await self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag_query_generation", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        print(f"RAG queries: {queries}")
        # retreive json triple backticks
        
        try: # add try-except block to handle potential json decode errors
            queries = re.search(r'```json(.*)```', queries, re.DOTALL).group(1)
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for storyboard: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    async def _generate_rag_queries_error_fix(self, error: str, code: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries for fixing code errors.

        Args:
            error (str): The error message to fix
            code (str): The code containing the error
            scene_trace_id (str, optional): Trace ID for the scene. Defaults to None.
            topic (str, optional): Topic of the scene. Defaults to None.
            scene_number (int, optional): Scene number. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list.

        Returns:
            List[str]: List of generated RAG queries for error fixing
        """
        # Create a cache key for this scene and error
        cache_key = f"{topic}_scene{scene_number}_error_fix"

        # Check if we already have a cache file for error fix queries
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_error_fix.json")

        # If cache file exists, load and return cached queries
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_queries = json.load(f)
                print(f"Using cached RAG queries for error fix in {cache_key}")
                return cached_queries

        # Generate new queries for error fix if not cached
        prompt = get_prompt_rag_query_generation_fix_error(
            error=error,
            code=code,
            relevant_plugins=", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        )

        queries = await self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag-query-generation-fix-error", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        # remove json triple backticks
        queries = queries.replace("```json", "").replace("```", "")
        try: # add try-except block to handle potential json decode errors
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for error fix: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    async def _load_scene1_reference_code(self, topic: str, file_prefix: str) -> Optional[str]:
        """Load the successfully rendered Scene 1 code as a reference.

        Args:
            topic (str): Topic name
            file_prefix (str): File prefix for the topic

        Returns:
            Optional[str]: Scene 1 code if found, None otherwise
        """
        import re
        import asyncio

        scene1_dir = os.path.join(self.output_dir, file_prefix, "scene1")
        scene1_code_dir = os.path.join(scene1_dir, "code")

        # Wait for Scene 1 to be successfully rendered (with timeout)
        max_wait_time = 600  # Maximum 10 minutes wait
        wait_interval = 5    # Check every 5 seconds
        total_waited = 0

        succ_marker = os.path.join(scene1_dir, "succ_rendered.txt")
        done_marker = os.path.join(scene1_code_dir, "scene1_code_tokens.json")

        while not os.path.exists(succ_marker) and total_waited < max_wait_time:
            # scene1_code_tokens.json exists means scene1 processing is done
            if os.path.exists(done_marker):
                raise RuntimeError(f"Scene 1 processing finished but render failed for {file_prefix}, skipping topic")
            if total_waited == 0:
                print(f"⏳ Waiting for Scene 1 to be successfully rendered before generating scene code...")
            else:
                print(f"⏳ Still waiting for Scene 1... ({total_waited}s elapsed)")

            await asyncio.sleep(wait_interval)  # Use async sleep instead of blocking time.sleep
            total_waited += wait_interval

        if not os.path.exists(succ_marker):
            print(f"⚠️  Scene 1 not successfully rendered after {max_wait_time}s, proceeding without reference")
            return None

        print(f"✓ Scene 1 successfully rendered after {total_waited}s, loading reference code")

        # Read the successful version number
        try:
            with open(succ_marker, 'r') as f:
                version_str = f.read().strip()
                # Extract version number (format: "v2" or just "2")
                if version_str.startswith('v'):
                    version_num = int(version_str[1:])
                else:
                    version_num = int(version_str)
        except:
            version_num = 0

        # Find the code file
        code_dir = os.path.join(scene1_dir, "code")
        code_file = os.path.join(code_dir, f"{file_prefix}_scene1_v{version_num}.py")

        if not os.path.exists(code_file):
            # Try to find any successfully rendered version
            if os.path.isdir(code_dir):
                code_files = [f for f in os.listdir(code_dir) if f.endswith('.py') and 'scene1' in f]
                if code_files:
                    # Sort by version number and take the latest
                    code_files.sort(key=lambda x: int(re.search(r'_v(\d+)\.py$', x).group(1)) if re.search(r'_v(\d+)\.py$', x) else 0)
                    code_file = os.path.join(code_dir, code_files[-1])
                else:
                    return None
            else:
                return None

        # Read and return the code
        try:
            with open(code_file, 'r') as f:
                code = f.read()
                print(f"✓ Loaded Scene 1 reference code from: {code_file}")
                return code
        except Exception as e:
            print(f"❌ Failed to load Scene 1 code: {e}")
            return None

    async def _extract_code_with_retries(self, response_text: str, pattern: str, generation_name: str = None, trace_id: str = None, session_id: str = None, max_retries: int = 10) -> str:
        """Extract code from response text with retry logic.

        Args:
            response_text (str): The text containing code to extract
            pattern (str): Regex pattern for extracting code
            generation_name (str, optional): Name of generation step. Defaults to None.
            trace_id (str, optional): Trace identifier. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            max_retries (int, optional): Maximum number of retries. Defaults to 10.

        Returns:
            str: The extracted code

        Raises:
            ValueError: If code extraction fails after max retries
        """
        retry_prompt = """
        Please extract the Python code in the correct format using the pattern: {pattern}. 
        You MUST NOT include any other text or comments. 
        You MUST return the exact same code as in the previous response, NO CONTENT EDITING is allowed.
        Previous response: 
        {response_text}
        """

        for attempt in range(max_retries):
            # Reasoning models (MiniMax-M3, DeepSeek, o3) wrap their thinking in
            # <think>...</think> and scatter illustrative ```python snippets inside it. Strip
            # those so they can't be mistaken for the program.
            cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
            # Prefer the most complete fenced ```python block. A non-greedy findall + longest
            # avoids the previous greedy `(.*)` span that glued every snippet/prose line between
            # the first and last fence onto the real program and broke it.
            blocks = re.findall(r'```python\s*\n?(.*?)```', cleaned, re.DOTALL)
            if blocks:
                return max(blocks, key=len).rstrip('\n')
            code_match = re.search(pattern, cleaned, re.DOTALL)
            if code_match:
                return code_match.group(1)

            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: Failed to extract code pattern. Retrying...")
                # Regenerate response with a more explicit prompt
                response_text = await self.scene_model(
                    _prepare_text_inputs(retry_prompt.format(pattern=pattern, response_text=response_text)),
                    metadata={
                        "generation_name": f"{generation_name}_format_retry_{attempt + 1}",
                        "trace_id": trace_id,
                        "session_id": session_id
                    }
                )
        
        raise ValueError(f"Failed to extract code pattern after {max_retries} attempts. Pattern: {pattern}")

    async def generate_manim_code(self,
                            topic: str,
                            description: str,
                            scene_outline: str,
                            scene_implementation: str,
                            scene_number: int,
                            additional_context: Union[str, List[str]] = None,
                            scene_trace_id: str = None,
                            session_id: str = None,
                            rag_queries_cache: Dict = None,
                            problem_image: Union[Image.Image, None] = None,
                            file_prefix: str = None) -> str:
        """Generate Manim code from explanation plan.

        Args:
            topic (str): Topic of the scene
            description (str): Description of the scene
            scene_outline (str): Outline of the scene
            scene_implementation (str): Implementation details
            scene_number (int): Scene number
            additional_context (Union[str, List[str]], optional): Additional context. Defaults to None.
            scene_trace_id (str, optional): Trace identifier. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            rag_queries_cache (Dict, optional): Cache for RAG queries. Defaults to None.
            problem_image (Image.Image, optional): Problem diagram image. Defaults to None.
            file_prefix (str, optional): File prefix for finding scene1 code. Defaults to None.

        Returns:
            Tuple[str, str]: Generated code and response text
        """
        if self.use_context_learning:
            # Add context examples to additional_context
            if additional_context is None:
                additional_context = []
            elif isinstance(additional_context, str):
                additional_context = [additional_context]

            # Now using the properly formatted code examples
            if self.context_examples:
                additional_context.append(self.context_examples)

        # Add Scene1 reference code for consistency (scene2 onwards)
        if scene_number > 1 and file_prefix:
            scene1_code = await self._load_scene1_reference_code(topic, file_prefix)
            if scene1_code:
                if additional_context is None:
                    additional_context = []
                scene1_reference = f"""
**Reference Code from Scene 1** (for style consistency):
Use the same visual style, color scheme, and layout principles as Scene 1.

```python
{scene1_code}
```

IMPORTANT: Maintain consistency with Scene 1:
- Use the same background color
- Use the same stroke widths and font sizes
- Use similar object positioning and spacing
- Follow the same code structure and organization
"""
                additional_context.append(scene1_reference)
                print(f"✓ Including Scene 1 reference code for consistency in scene {scene_number}")

        if self.use_rag:
            # Generate RAG queries (will use cache if available)
            rag_queries = await self._generate_rag_queries_code(
                implementation=scene_implementation,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id
            )

            retrieved_docs = self.vector_store.find_relevant_docs(
                queries=rag_queries,
                k=2, # number of documents to retrieve
                trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number
            )
            # Format the retrieved documents into a string
            if additional_context is None:
                additional_context = []
            additional_context.append(retrieved_docs)

        # Format code generation prompt with plan and retrieved context
        prompt = get_prompt_code_generation(
            scene_outline=scene_outline,
            scene_implementation=scene_implementation,
            topic=topic,
            description=description,
            scene_number=scene_number,
            additional_context=additional_context
        )

        # Prepare input with image if available
        if problem_image and scene_number == 1:
            messages = [
                {"type": "text", "content": prompt},
                {"type": "image", "content": problem_image}
            ]
            print(f"Including problem diagram in code generation for scene {scene_number}")
        else:
            messages = _prepare_text_inputs(prompt)

        # Generate code using model
        response_text = await self.scene_model(
            messages,
            metadata={"generation_name": "code_generation", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        # Extract code with retries
        code = await self._extract_code_with_retries(
            response_text,
            r"```python(.*)```",
            generation_name="code_generation",
            trace_id=scene_trace_id,
            session_id=session_id
        )
        return code, response_text

    async def fix_code_errors(self, implementation_plan: str, code: str, error: str, scene_trace_id: str, topic: str, scene_number: int, session_id: str, rag_queries_cache: Dict = None) -> str:
        """Fix errors in generated Manim code.

        Args:
            implementation_plan (str): Original implementation plan
            code (str): Code containing errors
            error (str): Error message to fix
            scene_trace_id (str): Trace identifier
            topic (str): Topic of the scene
            scene_number (int): Scene number
            session_id (str): Session identifier
            rag_queries_cache (Dict, optional): Cache for RAG queries. Defaults to None.

        Returns:
            Tuple[str, str]: Fixed code and response text
        """
        # Format error fix prompt
        prompt = get_prompt_fix_error(implementation_plan=implementation_plan, manim_code=code, error=error)

        if self.use_rag:
            # Generate RAG queries for error fixing
            rag_queries = await self._generate_rag_queries_error_fix(
                error=error,
                code=code,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id
            )
            retrieved_docs = self.vector_store.find_relevant_docs(
                queries=rag_queries,
                k=2, # number of documents to retrieve for error fixing
                trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number
            )
            # Format the retrieved documents into a string
            prompt = get_prompt_fix_error(implementation_plan=implementation_plan, manim_code=code, error=error, additional_context=retrieved_docs)

        # Get fixed code from model
        response_text = await self.scene_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "code_fix_error", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        # Extract fixed code with retries
        fixed_code = await self._extract_code_with_retries(
            response_text,
            r"```python(.*)```",
            generation_name="code_fix_error",
            trace_id=scene_trace_id,
            session_id=session_id
        )
        return fixed_code, response_text

    async def visual_self_reflection(self, code: str, media_path: Union[str, Image.Image], scene_trace_id: str, topic: str, scene_number: int, session_id: str, implementation: str = "", problem_image: Union[Image.Image, None] = None) -> str:
        """Use snapshot image or mp4 explanation to fix code.

        Args:
            code (str): Code to fix
            media_path (Union[str, Image.Image]): Path to media file or PIL Image
            scene_trace_id (str): Trace identifier
            topic (str): Topic of the scene
            scene_number (int): Scene number
            session_id (str): Session identifier
            implementation (str, optional): Implementation plan. Defaults to "".
            problem_image (Image.Image, optional): Original problem diagram for comparison. Defaults to None.

        Returns:
            Tuple[str, str]: Fixed code and response text
        """

        # Determine if we're dealing with explanation or image
        is_explanation = isinstance(media_path, str) and media_path.endswith('.mp4')

        # Load prompt template
        with open('task_generator/prompts_raw/prompt_visual_self_reflection.txt', 'r') as f:
            prompt_template = f.read()

        # Format prompt
        prompt = prompt_template.format(
            implementation=implementation,
            generated_code=code
        )


        # Prepare input based on media type
        if is_explanation and isinstance(self.scene_model, (GeminiWrapper, VertexAIWrapper)):
            # For explanation with Gemini models
            messages = [
                {"type": "text", "content": prompt},
                {"type": "explanation", "content": media_path}
            ]
            # Add problem image if available
            if problem_image:
                messages.append({
                "type": "text", 
                "content": "**[Original Problem Diagram]** The following is the original diagram from the problem statement:"
                })
                messages.append({"type": "image", "content": problem_image})
                print(f"Including problem diagram in visual self-reflection for scene {scene_number}")
        else:
            # For images or non-Gemini models
            if isinstance(media_path, str):
                media = Image.open(media_path)
            else:
                media = media_path
            messages = [
                {"type": "text", "content": prompt},
                {"type": "image", "content": media}
            ]
            # Add problem image if available
            if problem_image:
                messages.append({
                "type": "text", 
                "content": "**[Original Problem Diagram]** The following is the original diagram from the problem statement:"
                })
                messages.append({"type": "image", "content": problem_image})
                print(f"Including problem diagram in visual self-reflection for scene {scene_number}")
        
        # Get model response
        response_text = await self.scene_model(
            messages,
            metadata={
                "generation_name": "visual_self_reflection",
                "trace_id": scene_trace_id,
                "tags": [topic, f"scene{scene_number}"],
                "session_id": session_id
            }
        )

        # Check for <LGTM> or banned reasonings BEFORE attempting code extraction
        # If found, return original code unchanged (diagram is approved)
        if "<LGTM>" in response_text or any(word in response_text for word in self.banned_reasonings):
            print(f"Visual self-reflection: diagram approved (LGTM or termination keyword found)")
            return code, response_text

        # Extract code with retries (only if not <LGTM>)
        try:
            fixed_code = await self._extract_code_with_retries(
                response_text,
                r"```python(.*)```",
                generation_name="visual_self_reflection",
                trace_id=scene_trace_id,
                session_id=session_id
            )
            return fixed_code, response_text
        except ValueError as e:
            # If code extraction fails, return original code to avoid empty files
            print(f"Warning: Visual fix code extraction failed: {e}. Keeping original code.")
            return code, response_text