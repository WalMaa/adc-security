from lmstudio import call_lm_studio
from csvutils import save_to_csv
from models import call_transformer

# Example scenarios to analyze
scenarios = [
    {"ID": "V1", "DESCRIPTION": "Communication channels not adequately protected"},
    {"ID": "V2", "DESCRIPTION": "Uncontrolled changes to the operating system"}
]

# analysis_results = call_lm_studio(scenarios, "deepseek-r1-distill-qwen-7b")

analysis_results = call_transformer(scenarios[0]["DESCRIPTION"], "facebook/opt-125m")




print(analysis_results)
# Save the analysis results to a CSV file
# save_to_csv(analysis_results, "analysis_results.csv")

