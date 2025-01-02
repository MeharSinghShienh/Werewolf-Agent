'''
Experimental parent base parser, currently incomplete and unused
'''

"""The base class for model response parser."""
from abc import ABC, abstractmethod
from typing import Union, Sequence

from loguru import logger

from agentscope.exception import TagNotFoundError
from agentscope.models import ModelResponse

# TODO: Support one-time warning in logger rather than setting global variable
_FIRST_TIME_TO_REPORT_CONTENT = True
_FIRST_TIME_TO_REPORT_MEMORY = True


class ParserBase(ABC):
    """The base class for model response parser."""

    @abstractmethod
    def parse(self, response: ModelResponse) -> ModelResponse:
        """Parse the response text to a specific object, and stored in the
        parsed field of the response object."""

    def _extract_first_content_by_tag(
        self,
        response: ModelResponse,
        tag_start: str,
        tag_end: str,
    ) -> str:
        """Extract the first text content between the tag_start and tag_end
        in the response text. Note this function does not support nested.

        Args:
            response (`ModelResponse`):
                The response object.
            tag_start (`str`):
                The start tag.
            tag_end (`str`):
                The end tag.

        Returns:
            `str`: The extracted text content.
        """
        #text = response.text
        text = response

        index_start = text.find(tag_start)

        # Avoid the case that tag_begin contains tag_end, e.g. ```json and ```
        if index_start == -1:
            index_end = text.find(tag_end, 0)
        else:
            index_end = text.find(tag_end, index_start + len(tag_start))

        if index_start == -1 or index_end == -1:
            missing_tags = []
            if index_start == -1:
                missing_tags.append(tag_start)
            if index_end == -1:
                missing_tags.append(tag_end)

            raise TagNotFoundError(
                f"Missing "
                f"tag{'' if len(missing_tags) == 1 else 's'} "
                f"{' and '.join(missing_tags)} in response: {text}",
                raw_response=text,
                missing_begin_tag=index_start == -1,
                missing_end_tag=index_end == -1,
            )

        extract_text = text[
            index_start + len(tag_start) : index_end  # noqa: E203
        ]

        return extract_text