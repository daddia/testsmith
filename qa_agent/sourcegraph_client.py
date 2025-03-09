"""
Sourcegraph integration module.

This module provides functionality to integrate with Sourcegraph's APIs
for enhanced code context gathering and code intelligence.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import requests

from qa_agent.config import QAAgentConfig
from qa_agent.error_recovery import CircuitBreaker, ErrorHandler
from qa_agent.models import CodeIntelligenceResult, CodeSearchResult
from qa_agent.utils.logging import log_exception

logger = logging.getLogger(__name__)


class SourcegraphClient:
    """Client for interacting with Sourcegraph APIs."""

    def __init__(self, config: QAAgentConfig):
        """
        Initialize the Sourcegraph client.

        Args:
            config: Configuration object with Sourcegraph settings
        """
        self.api_endpoint = config.sourcegraph_api_endpoint
        self.api_token = config.sourcegraph_api_token
        self.limited_mode = False

        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}

        # Initialize error handler with standardized configuration
        self.error_handler = ErrorHandler(max_retries=3, backoff_factor=1.5)

        # Initialize circuit breaker for API requests
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        if self.api_token:
            self.headers["Authorization"] = f"token {self.api_token}"
            logger.info("Sourcegraph client initialized with API token")
        else:
            self.limited_mode = True
            logger.warning("Sourcegraph client initialized in limited mode (no API token)")
            logger.warning("Some features may be restricted or unavailable")

    def is_authenticated(self) -> bool:
        """
        Check if the client is authenticated with an API token.

        Returns:
            True if authenticated with token, False otherwise
        """
        return not self.limited_mode

    def search_code(self, query: str, limit: int = 10) -> List[CodeSearchResult]:
        """
        Search code using Sourcegraph's search API.

        Args:
            query: The search query string
            limit: Maximum number of results to return

        Returns:
            List of CodeSearchResult objects
        """
        logger.info(f"Searching Sourcegraph with query: {query}")

        # Check if circuit breaker allows execution
        if not self.circuit_breaker.can_execute("sourcegraph_search"):
            logger.warning("Circuit breaker is open. Skipping Sourcegraph API call.")
            return []

        def _execute_search() -> List[CodeSearchResult]:
            """Internal function to execute search with error handling"""
            url = f"{self.api_endpoint}/search/stream"
            params = {"q": query, "limit": limit}

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            results = []
            # Parse NDJSON response
            for line in response.text.splitlines():
                if line.strip():
                    try:
                        result_json = json.loads(line)
                        if "type" in result_json and result_json["type"] == "match":
                            match = result_json.get("data", {})
                            file_path = match.get("path", "")
                            repository = match.get("repository", "")
                            content = match.get("content", "")
                            line_start = match.get("lineNumber", 0)
                            line_end = line_start + len(match.get("lineMatches", []))
                            commit = match.get("commit", "")

                            # Construct URL to the result on Sourcegraph
                            url = f"{self.api_endpoint.replace('/.api', '')}/{repository}/-/blob/{commit}/{file_path}"

                            # Extract snippets
                            snippets = []
                            for line_match in match.get("lineMatches", []):
                                if "line" in line_match:
                                    snippets.append(line_match["line"])

                            result = CodeSearchResult(
                                file_path=file_path,
                                repository=repository,
                                content=content,
                                line_start=line_start,
                                line_end=line_end,
                                commit=commit,
                                url=url,
                                snippets=snippets,
                            )
                            results.append(result)
                    except json.JSONDecodeError as json_err:
                        logger.warning(
                            f"Failed to parse JSON from Sourcegraph response line: {line}"
                        )
                        log_exception(logger, "search_code", json_err, {"line": line})

            self.circuit_breaker.record_success("sourcegraph_search")
            logger.info(f"Found {len(results)} code search results")
            return results

        try:
            # Use error handler with retry mechanism
            results = self.error_handler.execute_with_retry(
                _execute_search,
                operation_name="sourcegraph_search_code",
                diagnostic_info={"query": query, "limit": limit},
                recoverable_exceptions=[
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.HTTPError,
                ],
            )
            return results
        except Exception as e:
            # Record failure in circuit breaker
            self.circuit_breaker.record_failure("sourcegraph_search")
            log_exception(logger, "search_code", e, {"query": query, "limit": limit})
            logger.error(f"Error searching Sourcegraph code: {str(e)}")
            return []

    def semantic_search(self, query: str, limit: int = 10) -> List[CodeSearchResult]:
        """
        Perform semantic code search using Sourcegraph's semantic search API.

        Args:
            query: The natural language query
            limit: Maximum number of results to return

        Returns:
            List of CodeSearchResult objects
        """
        logger.info(f"Performing semantic search with query: {query}")

        # Modify the query to use Sourcegraph's semantic search syntax if available
        semantic_query = f"content:{query} patternType:semantic"

        return self.search_code(semantic_query, limit)

    def get_code_intelligence(
        self, file_path: str, line: int, character: int = 0
    ) -> Optional[CodeIntelligenceResult]:
        """
        Get code intelligence information for a specific location.

        Args:
            file_path: Path to the file
            line: Line number (0-based)
            character: Character position (0-based)

        Returns:
            CodeIntelligenceResult object or None if request failed
        """
        logger.info(f"Getting code intelligence for {file_path} at line {line}")

        try:
            url = f"{self.api_endpoint}/lsif/hover"

            # Extract repo and path from file_path
            # Assuming file_path is in format "repo/path"
            parts = file_path.split("/", 1)
            if len(parts) < 2:
                logger.error(f"Invalid file path format: {file_path}")
                return None

            repo, path = parts

            params = {"repository": repo, "path": path, "line": line, "character": character}

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            data = response.json()

            # Get definitions
            definitions_url = f"{self.api_endpoint}/lsif/definitions"
            definitions_response = requests.get(
                definitions_url, headers=self.headers, params=params
            )
            definitions_data = (
                definitions_response.json()
                if definitions_response.status_code == 200
                else {"locations": []}
            )

            # Get references
            references_url = f"{self.api_endpoint}/lsif/references"
            references_response = requests.get(references_url, headers=self.headers, params=params)
            references_data = (
                references_response.json()
                if references_response.status_code == 200
                else {"locations": []}
            )

            result = CodeIntelligenceResult(
                definitions=definitions_data.get("locations", []),
                references=references_data.get("locations", []),
                hover_info=data.get("markdown", {}).get("text", ""),
                type_info=data.get("markdown", {}).get("kind", ""),
            )
            self.circuit_breaker.record_success("code_intelligence")
            return result

        except requests.RequestException as e:
            self.circuit_breaker.record_failure("code_intelligence")
            log_exception(
                logger, "get_code_intelligence", e, {"file_path": file_path, "line": line}
            )
            logger.error(f"Error getting code intelligence: {str(e)}")
            return None

    def get_scip_index(self, repository: str) -> Dict[str, Any]:
        """
        Get SCIP index data for a repository.

        Args:
            repository: The repository name

        Returns:
            Dictionary with SCIP index data
        """
        logger.info(f"Getting SCIP index for repository: {repository}")

        try:
            url = f"{self.api_endpoint}/git/blobs/{quote(repository)}/HEAD/.scip/index.scip"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            self.circuit_breaker.record_success("scip_index")
            return {"status": "success", "bytes": len(response.content)}

        except requests.RequestException as e:
            self.circuit_breaker.record_failure("scip_index")
            log_exception(logger, "get_scip_index", e, {"repository": repository})
            logger.error(f"Error getting SCIP index: {str(e)}")
            return {"status": "error", "message": str(e)}

    def find_examples(self, function_name: str, limit: int = 5) -> List[CodeSearchResult]:
        """
        Find usage examples of a function across repositories.

        Args:
            function_name: Name of the function to find examples for
            limit: Maximum number of examples to return

        Returns:
            List of CodeSearchResult objects with examples
        """
        logger.info(f"Finding examples for function: {function_name}")

        # Check if we're in limited mode
        if self.limited_mode:
            logger.warning("Limited mode: function example search may be restricted")

        # Construct a query that looks for function calls
        # In limited mode, use a simpler query that might work without authentication
        if self.limited_mode:
            query = f"{function_name}"
        else:
            query = f"\\b{function_name}\\([^)]*\\) patternType:regexp"

        def _execute_search():
            """Inner function to execute the function example search."""
            return self.search_code(query, limit)

        try:
            # Use error handler with retry mechanism
            results = self.error_handler.execute_with_retry(
                _execute_search,
                operation_name="find_examples",
                diagnostic_info={"function_name": function_name, "limit": limit},
                recoverable_exceptions=[
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.HTTPError,
                ],
            )
            self.circuit_breaker.record_success("function_examples")
            return results
        except Exception as e:
            # Record failure in circuit breaker
            self.circuit_breaker.record_failure("function_examples")
            log_exception(logger, "find_examples", e, {"function_name": function_name})
            logger.warning(f"Function example search failed: {str(e)}")

            # Try a more basic search as fallback
            basic_query = function_name
            return self.search_code(basic_query, limit)

    def find_related_code(self, code_snippet: str, limit: int = 5) -> List[CodeSearchResult]:
        """
        Find code related to a specific code snippet using semantic search.

        Args:
            code_snippet: The code snippet to find related code for
            limit: Maximum number of results to return

        Returns:
            List of CodeSearchResult objects
        """
        logger.info(f"Finding code related to snippet")

        # Check if we're in limited mode
        if self.limited_mode:
            logger.warning("Limited mode: semantic search may be restricted")

        # Use semantic search if available, otherwise fall back to keyword search
        def _execute_search():
            """Inner function to execute the search with appropriate mode."""
            if self.limited_mode:
                # Extract keywords from the code snippet
                keywords = self._extract_keywords(code_snippet)
                query = " ".join(keywords)
                return self.search_code(query, limit)
            else:
                return self.semantic_search(code_snippet, limit)

        try:
            # Use error handler with retry mechanism
            results = self.error_handler.execute_with_retry(
                _execute_search,
                operation_name="find_related_code",
                diagnostic_info={"snippet_length": len(code_snippet), "limit": limit},
                recoverable_exceptions=[
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.HTTPError,
                ],
            )
            self.circuit_breaker.record_success("related_code_search")
            return results
        except Exception as e:
            # Record failure in circuit breaker and fall back to keyword search
            self.circuit_breaker.record_failure("related_code_search")
            log_exception(logger, "find_related_code", e, {"snippet_length": len(code_snippet)})
            logger.warning(f"Search failed, falling back to basic keyword search: {str(e)}")

            # Extract keywords from the code snippet
            keywords = self._extract_keywords(code_snippet)
            query = " ".join(keywords)

            return self.search_code(query, limit)

    def _extract_keywords(self, code_snippet: str) -> List[str]:
        """
        Extract relevant keywords from a code snippet.

        Args:
            code_snippet: The code snippet

        Returns:
            List of keywords
        """
        # Simple keyword extraction - could be improved with NLP
        # Remove common symbols and split on whitespace
        cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in code_snippet)
        words = cleaned.split()

        # Filter out common keywords and short words
        stopwords = {
            "def",
            "if",
            "else",
            "for",
            "while",
            "return",
            "import",
            "from",
            "class",
            "and",
            "or",
            "not",
            "in",
            "is",
        }
        keywords = [word for word in words if word.lower() not in stopwords and len(word) > 2]

        return keywords[:10]  # Limit to 10 keywords
