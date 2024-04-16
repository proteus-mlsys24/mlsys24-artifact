import onnx
import pytest
from proteus.graph import Graph

@pytest.mark.parametrize("filename", ["/tmp/resnet18_1_3_224_224.onnx"])
def test_onnx_rules(filename: str):
    graph = Graph.from_onnx(filename)
    assert graph
