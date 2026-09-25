import streamlit as st
from diet import (
    bmi_calculator,
    bmr_calculator,
    tdee_calculator,
    calorie_target
)
from rag import load_rag
from openai import OpenAI


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="AI Health Assistant",
    layout="wide"
)


# --------------------------------------------------
# HUGGING FACE / LLM
# --------------------------------------------------

try:
    hf_token = st.secrets["HF_TOKEN"]

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token
    )

except Exception as e:
    client = None
    st.error(f"LLM configuration error: {e}")


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("AI HEALTH ASSISTANT 🏋️")

st.write(
    "Personal Health Assistance and Diet Recommendation Agent"
)

st.header("Health Information")


# --------------------------------------------------
# SIDEBAR - USER INFORMATION
# --------------------------------------------------

st.sidebar.header("🤷‍♂️ Your Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=20
)

weight = st.sidebar.number_input(
    "Weight(Kg)",
    min_value=1,
    max_value=120,
    value=60
)

height = st.sidebar.number_input(
    "Height(cm)",
    min_value=100,
    max_value=200,
    value=170
)

activity = st.sidebar.selectbox(
    "Activity",
    [
        "Sedentary",
        "Lightly Active",
        "Moderately Active",
        "Very Active",
        "Extra Active"
    ]
)

aim = st.sidebar.selectbox(
    "AIM",
    [
        "weight maintain",
        "weight loss",
        "weight gain"
    ]
)

diet_type = st.sidebar.selectbox(
    "Diet Type",
    [
        "Vegeterian",
        "Non Vegetarian"
    ]
)

allergies = st.sidebar.selectbox(
    "Allergies",
    [
        "None",
        "Allergies"
    ]
)


# --------------------------------------------------
# CALCULATIONS
# --------------------------------------------------

bmi = bmi_calculator(
    weight,
    height
)

bmr = bmr_calculator(
    gender,
    age,
    weight,
    height
)

tdee = tdee_calculator(
    bmr,
    activity
)

calories = calorie_target(
    tdee,
    aim
)


# --------------------------------------------------
# HEALTH INFORMATION DISPLAY
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "BMI",
    bmi
)

col2.metric(
    "BMR",
    f"{bmr} Kcal"
)

col3.metric(
    "TDEE",
    f"{tdee} Kcal"
)

col4.metric(
    "Calorie Target",
    f"{calories} Kcal"
)


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2 = st.tabs(
    [
        "Diet Recommendation",
        "Health Assistance"
    ]
)


# ==================================================
# TAB 1 - DIET RECOMMENDATION
# ==================================================

with tab1:

    if st.button("Recommend Diet"):

        if client is None:
            st.error(
                "LLM is not connected. Check HF_TOKEN in Streamlit Secrets."
            )

        else:

            with st.spinner("Creating Diet..."):

                try:

                    # Load RAG database
                    db = load_rag()

                    # Search query
                    search_query = f"""
                    Diet type: {diet_type}

                    Healthy food

                    Protein

                    Allergies: {allergies}
                    """

                    docs = db.similarity_search(
                        search_query,
                        k=3
                    )

                    # Create context
                    context = "\n\n".join(
                        [
                            doc.page_content
                            for doc in docs
                        ]
                    )

                    # Prompt
                    prompt = f"""
You are a helpful AI nutrition assistant.

Use the following nutrition knowledge
to create a simple one-day diet plan.

NUTRITION KNOWLEDGE:

{context}


USER INFORMATION:

Age: {age}

Gender: {gender}

Height: {height} cm

Weight: {weight} kg

Activity Level: {activity}

Aim: {aim}

Diet Type: {diet_type}

Food Allergy: {allergies}

Estimated BMI: {bmi}

Estimated BMR: {bmr} kcal/day

Estimated TDEE: {tdee} kcal/day

Estimated Daily Calorie Target:
{calories} kcal/day


Create the following:

1. Breakfast
2. Morning Snack
3. Lunch
4. Evening Snack
5. Dinner


For every meal provide:

- Food
- Portion
- Approximate calories
- Approximate protein


IMPORTANT RULES:

- Respect the user's diet type.
- Do not recommend foods containing the stated allergy.
- Use the provided nutrition knowledge when possible.
- Keep the plan simple and practical.
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not claim to cure diseases.
- This is general wellness information,
  not medical advice.
"""

                    # Call LLM
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    answer = response.choices[0].message.content

                    st.markdown(answer)

                except Exception as e:

                    # IMPORTANT:
                    # Show the real RAG error
                    st.error(
                        f"RAG Error: {e}"
                    )


# ==================================================
# TAB 2 - HEALTH ASSISTANCE
# ==================================================

with tab2:

    question = st.text_area(
        "Ask About Health",
        placeholder="e.g. Good source of vegetarian protein"
    )

    if st.button("Ask AI"):

        if not question.strip():

            st.warning(
                "Please enter a health question."
            )

        elif client is None:

            st.error(
                "LLM is not connected. Check HF_TOKEN in Streamlit Secrets."
            )

        else:

            with st.spinner("Thinking..."):

                try:

                    # Load RAG
                    db = load_rag()

                    # Search knowledge
                    docs = db.similarity_search(
                        question,
                        k=3
                    )

                    # Create context
                    context = "\n\n".join(
                        [
                            doc.page_content
                            for doc in docs
                        ]
                    )

                    # Prompt
                    prompt = f"""
You are an AI health and nutrition assistant.

Use the following knowledge to answer
the user's question.

NUTRITION KNOWLEDGE:

{context}


USER QUESTION:

{question}


INSTRUCTIONS:

- Answer clearly.
- Keep the explanation beginner-friendly.
- Use the provided knowledge when possible.
- Do not invent medical facts.
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not claim to cure diseases.
- If the question concerns a serious medical problem,
  recommend consulting a qualified healthcare professional.

This application provides general health
and nutrition information for educational
and wellness purposes.
"""

                    # Call LLM
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    answer = response.choices[0].message.content

                    st.markdown(answer)

                except Exception as e:

                    st.error(
                        f"RAG Error: {e}"
                    )


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.warning(
    "Knowledge-based health information tools serve as guides, "
    "but they cannot replace a physical examination or "
    "professional medical diagnosis."
)
