"""Tool descriptions and parameter descriptions in ``inputSchema``.

Covers:
- ``Annotated[T, Field(...)]`` metadata (description and constraints) reaching
  the schema, and enforced at call time
- ``Field(...)`` used as a parameter default
- Fallback to the docstring ``Args:`` section (Google and Sphinx styles)
- Precedence: a ``Field`` description wins over the docstring
- The summary paragraph as the tool description
- The same schema over a real stdio ``MCPClient`` connection
"""

from __future__ import annotations

import sys
import textwrap
from typing import Annotated, Optional

from pydantic import BaseModel, Field

from promptise.mcp.client import MCPClient
from promptise.mcp.server import MCPServer, TestClient
from promptise.mcp.server._decorators import (
    _docstring_summary,
    _extract_param_doc,
    _parse_param_docs,
)


class Address(BaseModel):
    """A postal address."""

    city: str = Field(description="City name")


async def _tool(server: MCPServer, name: str):
    tools = await TestClient(server).list_tools()
    return next(t for t in tools if t.name == name)


# =====================================================================
# Annotated / Field metadata
# =====================================================================


class TestAnnotatedField:
    async def test_annotated_description_reaches_schema(self):
        server = MCPServer(name="t")

        @server.tool()
        async def get_order_status(
            order_id: Annotated[str, Field(description='The order ID, for example "A-1001".')],
        ) -> dict:
            """Get the status of an order."""
            return {"order_id": order_id}

        tool = await _tool(server, "get_order_status")
        prop = tool.inputSchema["properties"]["order_id"]
        assert prop["description"] == 'The order ID, for example "A-1001".'
        assert prop["type"] == "string"
        assert tool.inputSchema["required"] == ["order_id"]

    async def test_annotated_constraints_reach_schema_and_are_enforced(self):
        server = MCPServer(name="t")

        @server.tool()
        async def list_orders(
            limit: Annotated[int, Field(ge=1, le=100, description="Max orders.")] = 10,
            status: Annotated[str, Field(pattern=r"^(open|closed)$")] = "open",
        ) -> dict:
            """List orders."""
            return {"limit": limit, "status": status}

        tool = await _tool(server, "list_orders")
        limit = tool.inputSchema["properties"]["limit"]
        assert limit["minimum"] == 1
        assert limit["maximum"] == 100
        assert limit["default"] == 10
        assert limit["description"] == "Max orders."
        assert tool.inputSchema["properties"]["status"]["pattern"] == r"^(open|closed)$"
        assert "required" not in tool.inputSchema

        client = TestClient(server)
        ok = await client.call_tool("list_orders", {"limit": 5})
        assert '"limit": 5' in ok[0].text
        bad = await client.call_tool("list_orders", {"limit": 0})
        assert "VALIDATION_ERROR" in bad[0].text
        bad = await client.call_tool("list_orders", {"status": "lost"})
        assert "VALIDATION_ERROR" in bad[0].text

    async def test_field_as_default(self):
        server = MCPServer(name="t")

        @server.tool()
        async def search(
            query: str = Field(description="Search text.", min_length=1),
            limit: int = Field(default=5, description="Max results."),
        ) -> dict:
            """Search."""
            return {"query": query, "limit": limit}

        tool = await _tool(server, "search")
        props = tool.inputSchema["properties"]
        assert props["query"]["description"] == "Search text."
        assert props["query"]["minLength"] == 1
        assert props["limit"] == {
            "default": 5,
            "description": "Max results.",
            "title": "Limit",
            "type": "integer",
        }
        assert tool.inputSchema["required"] == ["query"]

        result = await TestClient(server).call_tool("search", {"query": "x"})
        assert '"limit": 5' in result[0].text

    async def test_optional_annotated(self):
        server = MCPServer(name="t")

        @server.tool()
        async def find(
            customer: Annotated[Optional[str], Field(description="Customer ID.")] = None,
        ) -> dict:
            """Find."""
            return {"customer": customer}

        tool = await _tool(server, "find")
        prop = tool.inputSchema["properties"]["customer"]
        assert prop["description"] == "Customer ID."
        assert prop["default"] is None


# =====================================================================
# Docstring fallback
# =====================================================================


class TestDocstringArgs:
    async def test_google_args_reach_schema(self):
        server = MCPServer(name="t")

        @server.tool()
        async def get_order_status(order_id: str, include_items: bool = False) -> dict:
            """Get the status of an order.

            Args:
                order_id: The order ID, for example "A-1001". Case
                    sensitive.
                include_items (bool): Also return the line items.

            Returns:
                The order status.
            """
            return {}

        tool = await _tool(server, "get_order_status")
        props = tool.inputSchema["properties"]
        assert props["order_id"]["description"] == (
            'The order ID, for example "A-1001". Case sensitive.'
        )
        assert props["include_items"]["description"] == "Also return the line items."
        assert tool.description == "Get the status of an order."

    async def test_sphinx_params_reach_schema(self):
        server = MCPServer(name="t")

        @server.tool()
        async def refund(order_id: str, amount: float) -> dict:
            """Refund an order.

            :param order_id: The order to refund.
            :param float amount: Amount in EUR.
            :returns: The refund record.
            """
            return {}

        props = (await _tool(server, "refund")).inputSchema["properties"]
        assert props["order_id"]["description"] == "The order to refund."
        assert props["amount"]["description"] == "Amount in EUR."

    async def test_field_description_wins_over_docstring(self):
        server = MCPServer(name="t")

        @server.tool()
        async def lookup(
            a: Annotated[str, Field(description="From Field.")],
            b: str = Field(description="From default Field."),
            c: Annotated[int, Field(ge=0)] = 0,
        ) -> dict:
            """Lookup.

            Args:
                a: From docstring.
                b: From docstring.
                c: From docstring, constraints kept.
            """
            return {}

        props = (await _tool(server, "lookup")).inputSchema["properties"]
        assert props["a"]["description"] == "From Field."
        assert props["b"]["description"] == "From default Field."
        assert props["c"]["description"] == "From docstring, constraints kept."
        assert props["c"]["minimum"] == 0

    async def test_model_param_uses_param_doc_not_model_doc(self):
        server = MCPServer(name="t")

        @server.tool()
        async def ship(to: Address) -> dict:
            """Ship.

            Args:
                to: Where the parcel goes.
            """
            return {}

        prop = (await _tool(server, "ship")).inputSchema["properties"]["to"]
        assert prop["description"] == "Where the parcel goes."
        assert prop["properties"]["city"]["description"] == "City name"

    async def test_undocumented_params_have_no_description(self):
        server = MCPServer(name="t")

        @server.tool()
        async def ping(host: str) -> str:
            """Ping a host."""
            return host

        prop = (await _tool(server, "ping")).inputSchema["properties"]["host"]
        assert "description" not in prop

    async def test_excluded_params_are_not_documented_into_schema(self):
        from promptise.mcp.server import Depends, RequestContext

        def get_db() -> str:
            return "db"

        server = MCPServer(name="t")

        @server.tool()
        async def query(sql: str, ctx: RequestContext, db: str = Depends(get_db)) -> str:
            """Run a query.

            Args:
                sql: The statement.
                ctx: Injected.
                db: Injected.
            """
            return sql

        schema = (await _tool(server, "query")).inputSchema
        assert list(schema["properties"]) == ["sql"]
        assert schema["properties"]["sql"]["description"] == "The statement."


# =====================================================================
# Tool description
# =====================================================================


class TestToolDescription:
    async def test_wrapped_summary_paragraph_is_joined(self):
        server = MCPServer(name="t")

        @server.tool()
        async def get_order_status(order_id: str) -> dict:
            """Get the current status of an order, including the carrier and
            the expected delivery date.

            Implementation note: reads from the replica, so it can lag.

            Args:
                order_id: The order ID.
            """
            return {}

        tool = await _tool(server, "get_order_status")
        assert tool.description == (
            "Get the current status of an order, including the carrier and "
            "the expected delivery date."
        )

    async def test_summary_stops_at_section_header_without_blank_line(self):
        server = MCPServer(name="t")

        @server.tool()
        async def f(x: str) -> str:
            """Do the thing.
            Args:
                x: The input.
            """
            return x

        tool = await _tool(server, "f")
        assert tool.description == "Do the thing."
        assert tool.inputSchema["properties"]["x"]["description"] == "The input."

    async def test_explicit_description_wins(self):
        server = MCPServer(name="t")

        @server.tool(description="Explicit.")
        async def f(x: str) -> str:
            """Docstring summary."""
            return x

        assert (await _tool(server, "f")).description == "Explicit."

    async def test_no_docstring_falls_back_to_name(self):
        server = MCPServer(name="t")

        @server.tool()
        async def nameless(x: str) -> str:
            return x

        assert (await _tool(server, "nameless")).description == "nameless"


# =====================================================================
# Parser units
# =====================================================================


class TestParser:
    def test_summary_ignores_sphinx_fields(self):
        assert _docstring_summary("Do it.\n:param x: X.") == "Do it."

    def test_summary_empty_when_docstring_starts_with_section(self):
        assert _docstring_summary("Args:\n    x: X.") == ""

    def test_args_section_closes_on_dedent(self):
        doc = "Summary.\n\nArgs:\n    x: X.\n\nSome trailing prose: not a param.\n"
        assert _parse_param_docs(doc) == {"x": "X."}

    def test_stars_and_description_on_next_line(self):
        doc = "S.\n\nArgs:\n    *args: Positional.\n    **kwargs:\n        Keyword.\n"
        assert _parse_param_docs(doc) == {"args": "Positional.", "kwargs": "Keyword."}

    def test_other_sections_are_not_params(self):
        doc = "S.\n\nReturns:\n    x: not a param.\n\nRaises:\n    ValueError: bad.\n"
        assert _parse_param_docs(doc) == {}

    def test_extract_param_doc_still_works(self):
        doc = "S.\n\nArgs:\n    text: The text\n        to summarize.\n"
        assert _extract_param_doc(doc, "text") == "The text to summarize."
        assert _extract_param_doc(doc, "missing") is None

    async def test_prompt_arguments_get_multiline_descriptions(self):
        server = MCPServer(name="t")

        @server.prompt()
        async def summarize(text: str) -> str:
            """Summarize text.

            Args:
                text: The text
                    to summarize.
            """
            return text

        prompts = await TestClient(server).list_prompts()
        assert prompts[0].arguments[0].description == "The text to summarize."


# =====================================================================
# Real stdio round trip
# =====================================================================


_STDIO_SERVER = textwrap.dedent(
    '''
    from typing import Annotated

    from pydantic import Field

    from promptise.mcp.server import MCPServer

    server = MCPServer(name="orders")


    @server.tool()
    async def get_order_status(
        order_id: Annotated[str, Field(description='The order ID, for example "A-1001".')],
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> dict:
        """Get the status of an order
        and its shipping events.

        Args:
            limit: Max shipping events to return.
        """
        return {"order_id": order_id, "limit": limit}


    if __name__ == "__main__":
        server.run(transport="stdio")
    '''
)


class TestStdioRoundTrip:
    async def test_list_tools_over_stdio(self, tmp_path):
        script = tmp_path / "orders_server.py"
        script.write_text(_STDIO_SERVER)

        async with MCPClient(
            transport="stdio", command=sys.executable, args=[str(script)]
        ) as client:
            tools = await client.list_tools()

        tool = next(t for t in tools if t.name == "get_order_status")
        assert tool.description == "Get the status of an order and its shipping events."
        props = tool.inputSchema["properties"]
        assert props["order_id"]["description"] == 'The order ID, for example "A-1001".'
        assert props["limit"]["description"] == "Max shipping events to return."
        assert props["limit"]["minimum"] == 1
        assert props["limit"]["maximum"] == 50
