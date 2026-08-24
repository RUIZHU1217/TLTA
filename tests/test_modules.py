import torch

from models.hypergraph import HyperGraphConvolutionBlock, Hypergraph


def test_hgcb_shape_and_gradient():
    vertices = torch.randn(2, 6, 8, requires_grad=True)
    incidence = torch.tensor(
        [
            [[1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
            [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]],
        ],
        dtype=torch.float32,
    )
    graph = Hypergraph(vertices, incidence)
    output = HyperGraphConvolutionBlock(8, 12)(graph)
    assert output.shape == (2, 6, 12)
    output.sum().backward()
    assert vertices.grad is not None

