# components/fusion_adaptation.py
def fusion_adaptation(unsupervised_actions, supervised_actions):
    fused_actions = unsupervised_actions + supervised_actions
    return fused_actions
