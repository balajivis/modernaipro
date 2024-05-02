from mlflow.metrics.genai import EvaluationExample, make_genai_metric

professionalism_metric = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is tailored to the context and audience. It often involves avoiding overly casual language, slang, or colloquialisms, and instead using clear, concise, and respectful language"
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below "
        "are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in some informal professional settings."
        "- Score 3: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 4: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for business or academic settings. "
        "- Score 5: Language is excessively formal, respectful, and avoids casual elements. Appropriate for the most formal settings such as textbooks. "
    ),
    examples=[
        EvaluationExample(
            input="What is MLflow?",
            output=(
                "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps you track experiments, package your code and models, and collaborate with your team, making the whole ML workflow smoother. It's like your Swiss Army knife for machine learning!"
            ),
            score=2,
            justification=(
                "The response is written in a casual tone. It uses contractions, filler words such as 'like', and exclamation points, which make it sound less professional. "
            ),
        )
    ],
    version="v1",
    model="openai:/gpt-4",
    parameters={"temperature": 0.0},
    grading_context_columns=[],
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)
