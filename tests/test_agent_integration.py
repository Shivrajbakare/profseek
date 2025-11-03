from src.agent_utils import AgentOrchestrator
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_agent_runs():
    ag = AgentOrchestrator()
    summary = {"course": "EE220", "prof": "Dr. Example",
               "avg_grade": 8.2, "avg_star": 4.1, "reviews": []}
    out = ag.generate_course_advice(summary, "How is EE220?")
    assert out is not None
    print('Agent output:', out)
