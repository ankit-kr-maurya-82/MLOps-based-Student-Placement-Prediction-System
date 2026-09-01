const API_URL = "http://127.0.0.1:8000/predict";

const form = document.querySelector("#prediction-form");
const button = document.querySelector("#predict-button");
const formMessage = document.querySelector("#form-message");
const resultPanel = document.querySelector(".result-panel");
const resultTitle = document.querySelector("#result-title");
const resultLabel = document.querySelector("#result-label");
const resultDetail = document.querySelector("#result-detail");
const probabilityValue = document.querySelector("#probability-value");

const fieldLabels = {
  cgpa: "CGPA",
  tenth_percentage: "10th percentage",
  twelfth_percentage: "12th percentage",
  internships: "internships",
  projects: "projects",
  aptitude_score: "aptitude score",
  communication_score: "communication score",
  coding_score: "coding score",
  backlogs: "backlogs",
  certifications: "certifications",
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearMessage();

  const payload = readFormValues();
  const validationError = validatePayload(payload);

  if (validationError) {
    showMessage(validationError);
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(formatApiError(data));
    }

    renderResult(data);
  } catch (error) {
    showMessage(error.message || "Unable to reach the prediction API.");
  } finally {
    setLoading(false);
  }
});

function readFormValues() {
  return Object.fromEntries(
    Array.from(new FormData(form).entries()).map(([key, value]) => [
      key,
      Number(value),
    ])
  );
}

function validatePayload(payload) {
  const ranges = {
    cgpa: [0, 10],
    tenth_percentage: [0, 100],
    twelfth_percentage: [0, 100],
    internships: [0, 10],
    projects: [0, 20],
    aptitude_score: [0, 100],
    communication_score: [0, 100],
    coding_score: [0, 100],
    backlogs: [0, 20],
    certifications: [0, 20],
  };

  for (const [field, value] of Object.entries(payload)) {
    const [min, max] = ranges[field];

    if (!Number.isFinite(value)) {
      return `${fieldLabels[field]} is required.`;
    }

    if (value < min || value > max) {
      return `${fieldLabels[field]} must be between ${min} and ${max}.`;
    }
  }

  return "";
}

function renderResult(data) {
  const probability = Number(data.probability);
  const percentage = Math.round(probability * 100);
  const isPlaced = data.prediction === "Placed";

  resultPanel.classList.toggle("is-success", isPlaced);
  resultPanel.classList.toggle("is-risk", !isPlaced);
  resultPanel.style.setProperty("--meter-degrees", `${probability * 360}deg`);

  resultTitle.textContent = isPlaced ? "High Placement Chance" : "Needs Improvement";
  resultLabel.textContent = isPlaced ? "Placed" : "Not Placed";
  probabilityValue.textContent = `${percentage}%`;
  resultDetail.textContent = isPlaced
    ? "The profile shows strong placement readiness based on the trained model."
    : "The profile may need stronger academic, project, internship, or skill signals.";
}

function setLoading(isLoading) {
  button.disabled = isLoading;
  button.querySelector("span:last-child").textContent = isLoading
    ? "Predicting..."
    : "Predict Placement";
}

function showMessage(message) {
  formMessage.textContent = message;
}

function clearMessage() {
  formMessage.textContent = "";
}

function formatApiError(data) {
  if (typeof data?.detail === "string") {
    return data.detail;
  }

  if (Array.isArray(data?.detail) && data.detail[0]?.msg) {
    return data.detail[0].msg;
  }

  return "Prediction failed. Please check the inputs and try again.";
}
