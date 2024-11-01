## Extracting Data from PDFs with Azure OpenAI and GPT-4o: A Complete Example

In this blog post, we’ll explore building a pipeline to extract and analyze data from PDFs using the power of Microsoft Azure’s OpenAI service with GPT-4o. As a multimodal model, GPT-4o supports both text and image inputs, which makes it versatile for complex document processing tasks. By the end, you’ll know how to preprocess a PDF, interact with Azure OpenAI, and extract key information effectively.

---

### Why Azure OpenAI and GPT-4o?

Azure OpenAI’s GPT-4o model offers advanced capabilities:
- **Multimodal Data Handling**: Works with both text and images, enabling robust analysis.
- **Powerful Performance**: Matches GPT-4 Turbo in English text and coding tasks and performs well with non-English languages.
- **Scalable and Secure**: Leveraging Azure’s infrastructure, it provides scalability and compliance with data privacy standards.

---

### Example: Extracting Key Information from a Sample PDF

To illustrate the process, we’ll use a sample PDF containing business-related information:
  
```
Company: Acme Corp  
Location: San Francisco, CA  
Industry: Artificial Intelligence and Machine Learning Solutions  
Summary: Acme Corp specializes in developing advanced software solutions in AI, NLP, and computer vision.
  
Company: Beta Industries  
Location: New York, NY  
Industry: Sustainable Materials and Renewable Energy  
Summary: Focused on sustainable production, Beta Industries leads in innovative environmental technology.
```

---

### Step 1: Setting Up the Environment

#### Install the Required Libraries

```bash
pip install openai PyPDF2 python-dotenv
```

#### Load Azure OpenAI API Configuration

To connect to Azure OpenAI, we’ll load credentials and configurations from an environment file.

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("azure.env")

# Azure OpenAI API settings
api_type = "azure"
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
api_version = "2024-05-01-preview"
model = "gpt-4o"
```

#### Check OpenAI SDK Version

```python
import openai

def check_openai_version():
    print(f"OpenAI version: {openai.__version__}")
    if float(openai.__version__[:3]) < 1.0:
        print("Consider upgrading OpenAI to version >= 1.0.0")

check_openai_version()
```

---

### Step 2: Preprocessing PDFs

Convert PDF files into text format suitable for input into Azure OpenAI. Here, we use the `PyPDF2` library to extract text from the PDF.

```python
import PyPDF2

def preprocess_pdf(pdf_path):
    """
    Extracts and cleans text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Cleaned text from the PDF.
    """
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return " ".join(text.split())
```

---

### Step 3: Interacting with Azure OpenAI GPT-4o

With GPT-4o, you can analyze text to extract relevant information. Here, we’ll create a function to query the model for specific tasks, like identifying company names and descriptions.

```python
import openai

openai.api_type = api_type
openai.api_key = api_key
openai.api_base = api_base
openai.api_version = api_version

def gpt4o_text(prompt):
    """
    Sends a prompt to Azure OpenAI GPT-4o for text analysis.

    Args:
        prompt (str): The text prompt for GPT-4o.
    Returns:
        str: The model's response.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "Extract company names and descriptions."},
                  {"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message["content"]
```

---

### Step 4: Extracting Information

Define a structured prompt for data extraction, instructing the model to return a JSON-like output format to organize information about company names, locations, and industries.

```python
def extract_information(text):
    """
    Extracts structured company information using GPT-4o.

    Args:
        text (str): Cleaned text from the PDF.
    Returns:
        dict: Extracted company details.
    """
    prompt = f"""
    Identify and extract the company names, locations, and industries mentioned in the following text:
    {text}
    Provide output in the following JSON format:
    {{"companies": [
        {{"name": "Acme Corp", "location": "San Francisco", "industry": "AI"}},
        {{"name": "Beta Industries", "location": "New York", "industry": "Sustainable Materials"}}
    ]}}
    """
    response = gpt4o_text(prompt)
    return response
```

---

### Step 5: Putting It All Together

Combine all the steps to create a complete pipeline that extracts structured information from PDF files.

```python
def process_pdf(pdf_path):
    """
    Processes a PDF to extract company details.

    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        dict: Extracted information about companies.
    """
    text = preprocess_pdf(pdf_path)
    extracted_data = extract_information(text)
    return extracted_data

# Example usage
pdf_path = "example.pdf"
extracted_data = process_pdf(pdf_path)
print(extracted_data)
```

---

### Expected Output

Running the code will yield output similar to the following:

```json
{
  "companies": [
    {
      "name": "Acme Corp",
      "location": "San Francisco, CA",
      "industry": "Artificial Intelligence and Machine Learning Solutions",
      "summary": "Specializes in advanced software solutions in AI, NLP, and computer vision."
    },
    {
      "name": "Beta Industries",
      "location": "New York, NY",
      "industry": "Sustainable Materials and Renewable Energy",
      "summary": "Leads in innovative environmental technology."
    }
  ]
}
```

---

### Using GPT-4o for Image Analysis

To leverage GPT-4o’s multimodal capabilities, you can analyze images within PDFs, such as diagrams or embedded visuals, by sending the image data to Azure OpenAI.

```python
import base64
from io import BytesIO
from PIL import Image

def gpt4o_image_analysis(image_path, prompt):
    """
    Analyzes an image using GPT-4o.

    Args:
        image_path (str): Path to the image.
        prompt (str): Instruction for image analysis.
    Returns:
        str: Model's analysis.
    """
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")
    image_prompt = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
    }
    response = openai.ChatCompletion.create(
        model=model,
        messages=[image_prompt],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message["content"]
```

---

### Comparison with Other Models on Azure OpenAI

| Feature                | GPT-4o        | GPT-4 Turbo   |
|------------------------|---------------|---------------|
| **Language Support**   | Superior non-English handling | High performance |
| **Vision Capability**  | Yes           | No            |
| **Response Speed**     | Moderate      | High          |
| **Output Precision**   | High          | Very High     |
| **Typical Use Cases**  | Document analysis, multimodal tasks | Text and code-heavy applications |

### Conclusion

This Azure OpenAI-based pipeline, leveraging GPT-4o, showcases an efficient way to extract structured information from PDFs with text and images. The model’s flexibility and power make it a valuable tool for enterprise-level document processing and analysis. By adjusting prompts and configurations, you can further customize the pipeline to handle various document formats and complexities, from business reports to technical diagrams.