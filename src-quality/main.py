import sys
import json
from modules import APICatalogProject, CodeParsingAndAPIExtraction, APIDocumentationGeneration
from modules import APICatalogueCreation, APIRecommendationEngine, TestingAndValidation

if __name__ == "__main__":
    # 1️⃣ Initialize and Run the Project
    project = APICatalogProject("API Cataloguing", "Extract and document APIs")
    project.run()

    # 2️⃣ Extract APIs from a sample codebase
    extractor = CodeParsingAndAPIExtraction("sample_code/app.py")
    extracted_apis = extractor.extract_apis()
    print("Extracted APIs:", extracted_apis)

    # 3️⃣ Generate API documentation
    doc_generator = APIDocumentationGeneration(extracted_apis)
    api_docs = doc_generator.generate_documentation()
    print("Generated API Docs:", json.dumps(api_docs, indent=4))

    formatted_doc_summary = doc_generator.format_documentation()
    print(formatted_doc_summary)

    # 4️⃣ Store API documentation and build a search index
    catalog = APICatalogueCreation(api_docs)
    catalog.create_catalogue()
    catalog.build_search_index()
    catalog.save_to_db()

    # 5️⃣ Implement API Recommendation System
    search_engine = APIRecommendationEngine(catalog.search_index)
    query = "Retrieve user information"
    recommendation = search_engine.search(query)
    print("Recommended API:", json.dumps(recommendation, indent=4))

    # 6️⃣ Run Tests
    test_cases = [
        {
            "description": "Extract API from sample code",
            "function": extractor.extract_apis,
            "input": [],
            "expected": extracted_apis
        }
    ]
    validator = TestingAndValidation(test_cases)
    validator.run_tests()

    all_tests_passed = validator.validate_results()
    print(f"\n✅ All tests passed: {all_tests_passed}")
    if all_tests_passed:
        print(f"\n✅ All tests passed: {all_tests_passed}")
    else:
        print("\n❌ Some tests failed. Please check the test logs.")
        print("⚠️ Exiting program due to failed tests.")
        sys.exit(1)
