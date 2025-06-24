import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage

from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.analysts import get_analyst_nodes
from src.utils.progress import progress


class HedgeFundRunner:
    """Encapsulates the hedge fund workflow."""

    def __init__(self, selected_analysts: list[str] | None = None):
        self.workflow = create_workflow(selected_analysts)
        self.app = self.workflow.compile()

    def run(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        portfolio: dict,
        show_reasoning: bool = False,
        model_name: str = "gpt-4o",
        model_provider: str = "OpenAI",
    ) -> dict:
        """Execute the trading workflow."""
        progress.start()
        try:
            final_state = self.app.invoke(
                {
                    "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
                    "data": {
                        "tickers": tickers,
                        "portfolio": portfolio,
                        "start_date": start_date,
                        "end_date": end_date,
                        "analyst_signals": {},
                    },
                    "metadata": {
                        "show_reasoning": show_reasoning,
                        "model_name": model_name,
                        "model_provider": model_provider,
                    },
                }
            )

            final_message = final_state["messages"][-1]
            raw_response = getattr(final_message, "content", str(final_message))

            history = [
                {
                    "type": getattr(m, "type", type(m).__name__),
                    "name": getattr(m, "name", None),
                    "content": getattr(m, "content", ""),
                }
                for m in final_state.get("messages", [])
            ]

            return {
                "decisions": parse_hedge_fund_response(raw_response),
                "analyst_signals": final_state["data"]["analyst_signals"],
                "raw_response": raw_response,
                "messages": history,
            }
        finally:
            progress.stop()


def parse_hedge_fund_response(response: str) -> dict | None:
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
    return None


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts: list[str] | None = None) -> StateGraph:
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    analyst_nodes = get_analyst_nodes()
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())

    for key in selected_analysts:
        node_name, node_func = analyst_nodes[key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    for key in selected_analysts:
        node_name = analyst_nodes[key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)
    workflow.set_entry_point("start_node")
    return workflow


def default_dates(start_date: str | None, end_date: str | None) -> tuple[str, str]:
    """Return sane default start and end dates."""
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    if start_date:
        start = start_date
    else:
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        start = (end_dt - relativedelta(months=3)).strftime("%Y-%m-%d")
    return start, end
