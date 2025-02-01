from .models import generator
from .lmstudio import call_lm_studio
from .csvutils import save_to_csv

# Example scenarios to analyze
scenarios = [
    {"ID": "V1", "DESCRIPTION": "Communication channels not adequately protected"},
    {"ID": "V2", "DESCRIPTION": "Uncontrolled changes to the operating system"}
]

# analysis_results = call_lm_studio(scenarios, "deepseek-r1-distill-qwen-7b")
analysis_results = generator(scenarios)




print(analysis_results)
# Save the analysis results to a CSV file
# save_to_csv(analysis_results, "analysis_results.csv")

