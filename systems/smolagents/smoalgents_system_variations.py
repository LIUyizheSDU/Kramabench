from .smolagents_system import Smolagents

class SmolagentsDeepResearchGPTo3(Smolagents):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="gpt-o3",
            name="SmolagentsDeepResearchGPTo3",
            variance="DeepResearch",
            verbose=verbose,
            *args, **kwargs
        )
class SmolagentsPDTGPTo3(Smolagents):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="gpt-o3",
            name="SmolagentsPDTGPTo3",
            variance="PDT",
            verbose=verbose,
            *args, **kwargs
        )

class SmolagentsReflexionGPTo3(Smolagents):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="gpt-o3",
            name="SmolagentsReflexionGPTo3",
            variance="Reflexion",
            verbose=verbose,
            *args, **kwargs
        )

class SmolagentsDeepResearchClaude37(Smolagents):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="claude-37",
            name="SmolagentsDeepResearchClaude37",
            variance="DeepResearch",
            verbose=verbose,
            *args, **kwargs
        )

class SmolagentsPDTClaude37(Smolagents):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="claude-37",
            name="SmolagentsPDTClaude37",
            variance="PDT",
            verbose=verbose,
            *args, **kwargs
        )

class SmolagentsReflexionClaude37(Smolagents):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="claude-37",
            name="SmolagentsReflexionClaude37",
            variance="Reflexion",
            verbose=verbose,
            *args, **kwargs
        )