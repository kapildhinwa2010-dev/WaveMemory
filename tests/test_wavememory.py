import pytest
import torch
from wavememory import WaveMemory, create_wavememory

class TestWaveMemory:
    def test_init(self):
        m = WaveMemory(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = WaveMemory(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_wavememory()
        assert isinstance(m, WaveMemory)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
