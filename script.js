document.getElementById("upload-btn").addEventListener("click", async () => {
    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload a file first!");
        return;
    }

    // Show loading spinner
    document.getElementById("loader").style.display = "inline-block";
    document.getElementById("result-section").style.display = "none";

    // Form data for file upload
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://localhost:5000/detect", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to fetch results from the server');
        }

        const result = await response.json();

        // Hide loading spinner
        document.getElementById("loader").style.display = "none";

        // Display the result
        document.getElementById("result-section").style.display = "block";
        const resultMessage = result.prediction;
        const confidenceScore = (result.max_probability * 100).toFixed(2);

        document.getElementById("result-message").textContent = resultMessage;
        document.getElementById("confidence-score").textContent = `${confidenceScore}%`;

    } catch (error) {
        console.error("Error during deepfake detection:", error);
        document.getElementById("loader").style.display = "none";
        alert("An error occurred. Please try again later.");
    }
});
