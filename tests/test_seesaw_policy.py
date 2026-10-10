from dataclasses import replace
import unittest
import torch
from seesaw_policy.config import SensorConfig, memory_defaults
from seesaw_policy.model import SeesawPolicy
from seesaw_policy.train import returns


class PolicyTests(unittest.TestCase):
    def test_recurrent_replay_and_gradients(self):
        torch.manual_seed(1); torch.set_num_threads(2)
        sensor=SensorConfig()
        memory=memory_defaults()
        memory.update(dyn_deter=32,dyn_hidden=32,dyn_stoch=4,dyn_discrete=4)
        memory['encoder'].update(cnn_depth=4,mlp_layers=1,mlp_units=32)
        policy=SeesawPolicy(sensor,memory,7)
        state=policy.rssm.initial(2)
        initial={k:v.clone() for k,v in state.items()}
        obs=torch.randn(2,3,45)
        image=torch.rand(2,3,32,48,1)-.5
        previous=torch.randn(2,3,12)
        first=torch.tensor([[True,False,False],[True,False,True]])
        means=[]
        with torch.no_grad():
            for t in range(3):
                mean,state=policy.step(obs[:,t],policy.encode_image(image[:,t]),previous[:,t],first[:,t],state)
                means.append(mean)
        replay=policy.sequence(obs,image,previous,first,initial)
        torch.testing.assert_close(replay,torch.stack(means,1),atol=1e-6,rtol=1e-5)
        loss=replay.square().mean()+policy.value(obs,torch.randn(2,3,7)).square().mean()
        loss.backward()
        for module in (policy.encoder._cnn,policy.encoder._mlp,policy.rssm,policy.projector,policy.actor,policy.critic):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters()))
        self.assertTrue(all(p.requires_grad for p in policy.parameters()))
        restored=SeesawPolicy(sensor,memory,7)
        restored.load_state_dict(policy.state_dict(),strict=True)

    def test_task_keeps_relative_balance_and_only_seesaw(self):
        from seesaw_policy.task import make_task_recipe
        config=make_task_recipe().env_config['seesaw']
        self.assertEqual(config['reward_version'],4)
        self.assertEqual(config['reward_scales']['seesaw_level'],4.)
        self.assertEqual(config['reward_scales']['seesaw_stable'],0.)
        self.assertEqual(config['reward_scales']['seesaw_four_foot'],.2)
        self.assertEqual(config['seesaw_fraction'],1.)
        self.assertEqual(config['rough_fraction'],0.)
        self.assertEqual(config['flat_fraction'],0.)

    def test_terminal_and_timeout_returns(self):
        reward=torch.ones(1,2,2)
        value=torch.full_like(reward,2.)
        done=torch.ones(1,2,dtype=torch.bool)
        timeout=torch.tensor([[False,True]])
        result=returns(reward,value,done,timeout,torch.full((2,2),9.),gamma=.9)
        torch.testing.assert_close(result[0],torch.tensor([[1.,1.],[2.8,2.8]]))


if __name__=='__main__': unittest.main()
