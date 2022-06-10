import unittest

import torch

from models.networks import *

class Test_functions(unittest.TestCase):

    def test_cal_gradient_penalty(self):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        netD = NLayerDiscriminator(1)
        image_shape = (256, 1, 28, 28)


        netD = init_net(netD)
        
        real = torch.randn(*image_shape, device=device) + 1
        fake = torch.randn(*image_shape, device=device) - 1

        with self.subTest():
            # Test Gradient
            _, gradients = cal_gradient_penalty(netD, real, fake, device, type='mixed', constant=1.0, lambda_gp=10.0)

            #self.assertEqual(tuple(gradients.shape), image_shape, msg='gradient shape incorrect.')
            self.assertEqual(tuple(gradients.shape), (image_shape[0], image_shape[1]*image_shape[2]*image_shape[3]), msg='gradient shape incorrect.')
            self.assertGreater(gradients.max(), 0, msg="Max gradient not > 0.")
            self.assertLess(gradients.min(), 0, msg='Min gradient value not < 0.')

        with self.subTest():
            # Test Gradient penalty
            random_gradient_penalty, _ = cal_gradient_penalty(netD, real, fake, device, type='mixed', constant=1.0, lambda_gp=1)
            self.assertLess(torch.abs(random_gradient_penalty - 1), 0.1, msg='Random gradient penalty not < 0.1.')

"""
        # UNIT TEST
def test_gradient_penalty(image_shape):
    bad_gradient = torch.zeros(*image_shape)
    bad_gradient_penalty = gradient_penalty(bad_gradient)
    assert torch.isclose(bad_gradient_penalty, torch.tensor(1.))

    image_size = torch.prod(torch.Tensor(image_shape[1:]))
    good_gradient = torch.ones(*image_shape) / torch.sqrt(image_size)
    good_gradient_penalty = gradient_penalty(good_gradient)
    assert torch.isclose(good_gradient_penalty, torch.tensor(0.))

    random_gradient = test_get_gradient(image_shape)
    random_gradient_penalty = gradient_penalty(random_gradient)
    assert torch.abs(random_gradient_penalty - 1) < 0.1

test_gradient_penalty((256, 1, 28, 28))
print("Success!")

"""