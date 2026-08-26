"""Fast policy used before full MCTS search starts."""

INQUIRY_ACTIONS = ("GenreInquiry", "ActorInquiry", "DirectorInquiry")


def select_early_inquiry(action_scores, actions_taken=()):
    remaining = [action for action in INQUIRY_ACTIONS if action not in actions_taken]
    candidates = remaining or list(INQUIRY_ACTIONS)
    return max(candidates, key=lambda action: action_scores.get(action, 0))
