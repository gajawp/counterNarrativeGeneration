"""
Author: Preethi Gajawada
Institution: Northeastern University
Project: Multilingual Counter-Narrative Generation using DSPy
Description: Configures the base language model used by DSPy programs.
Year: 2026
"""

import dspy

def configure_gpt4o(model="gpt-4o-mini"):
    lm = dspy.LM(
        model=model,
        response_format={"type": "json_object"} 
    )
    dspy.settings.configure(lm=lm)
    return lm