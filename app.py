import streamlit as st
import boto3
import json

###############################################################################
# 1. Configuration - AWS Bedrock client
###############################################################################
# You must have AWS credentials properly configured for this to work.
# Replace 'us-east-1' with the region where you have Bedrock access.
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def call_llm(prompt: str, model="cohere.command-r-plus-v1:0", temperature=0.7) -> str:
    """
    Example LLM call using AWS Bedrock's Cohere model: cohere.command-r-plus-v1:0.
    Adjust the JSON structure based on the actual model's response in your environment.
    """
    try:
        body = {
            "prompt": prompt,
            "temperature": temperature,
        }

        response = bedrock.invoke_model(
            modelId=model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        response_body = response["body"].read()
        result = json.loads(response_body)

        # Adapt if the response structure changes
        if "completions" in result and len(result["completions"]) > 0:
            generated_text = result["completions"][0]["data"]["text"]
        else:
            generated_text = "No output from model."

        return generated_text.strip()

    except Exception as e:
        return f"Error calling Bedrock model: {e}"

###############################################################################
# 2. Streamlit UI
###############################################################################
st.title("Prompt Tester with AWS Bedrock")

# Prompt template input
st.subheader("Prompt Template")
prompt_template = st.text_area(
    "Enter your prompt with placeholders. (e.g. 'Hello, my name is {name}.')",
    value="Hello, my name is {name}. I love {hobby}."
)

# Placeholder values as JSON
st.subheader("Placeholder Values")
placeholder_json = st.text_area(
    "Enter your placeholder values in JSON format. (e.g. {\"name\": \"Alice\", \"hobby\": \"coding\"})",
    value='{"name": "Alice", "hobby": "coding"}'
)

# Model and temperature settings
model_name = st.text_input("Model Name (e.g., cohere.command-r-plus-v1:0)", "cohere.command-r-plus-v1:0")
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Generate button
if st.button("Generate"):
    # Safely parse JSON
    try:
        placeholders = json.loads(placeholder_json)
        # Inject placeholders into the prompt
        final_prompt = prompt_template.format(**placeholders)
    except Exception as e:
        st.error(f"Error parsing JSON or injecting variables: {e}")
        final_prompt = None

    if final_prompt:
        st.write("### Final Prompt:")
        st.code(final_prompt, language="markdown")
        
        # Call the LLM
        output = call_llm(
            prompt=final_prompt,
            model=model_name,
            temperature=temperature
        )
        
        st.write("### Model Output:")
        st.write(output)
