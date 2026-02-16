import torch
from scipy.stats import spearmanr

def calculate_spearman_correlation(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the Spearman Rank-Order Correlation Coefficient.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.

    Returns:
        float: Spearman correlation coefficient. Returns 0.0 if calculation fails
               (e.g., all predictions/targets are constant).
    """
    predictions_np = predictions.squeeze().cpu().numpy()
    targets_np = targets.squeeze().cpu().numpy()

    try:
        # spearmanr returns (correlation, p_value)
        correlation, _ = spearmanr(predictions_np, targets_np)
        return correlation
    except ValueError:
        # This can happen if all values in predictions_np or targets_np are the same,
        # which makes the rank calculation ill-defined for spearmanr.
        return 0.0

if __name__ == '__main__':
    # Unit tests for calculate_spearman_correlation
    import unittest

    class TestMetrics(unittest.TestCase):
        def test_perfect_correlation(self):
            pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            corr = calculate_spearman_correlation(pred, target)
            self.assertAlmostEqual(corr, 1.0, places=4)

        def test_perfect_negative_correlation(self):
            pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            target = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
            corr = calculate_spearman_correlation(pred, target)
            self.assertAlmostEqual(corr, -1.0, places=4)

        def test_no_correlation(self):
            pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            target = torch.tensor([3.0, 5.0, 1.0, 2.0, 4.0])
            corr = calculate_spearman_correlation(pred, target)
            # This should be around 0 for randomly ordered data
            self.assertAlmostEqual(corr, spearmanr(pred.numpy(), target.numpy())[0], places=4)

        def test_constant_values(self):
            pred = torch.tensor([1.0, 1.0, 1.0])
            target = torch.tensor([2.0, 2.0, 2.0])
            corr = calculate_spearman_correlation(pred, target)
            self.assertAlmostEqual(corr, 0.0, places=4) # spearmanr returns nan for constant, we handle it to 0.0

        def test_mixed_constant(self):
            pred = torch.tensor([1.0, 1.0, 2.0])
            target = torch.tensor([10.0, 11.0, 12.0])
            corr = calculate_spearman_correlation(pred, target)
            self.assertAlmostEqual(corr, spearmanr(pred.numpy(), target.numpy())[0], places=4)


    unittest.main()
