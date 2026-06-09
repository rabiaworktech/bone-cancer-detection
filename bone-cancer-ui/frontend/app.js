document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');

    const loadingState = document.getElementById('loadingState');
    const resultsSection = document.getElementById('resultsSection');
    const uploadArea = document.getElementById('uploadArea');

    const resultOriginalImage = document.getElementById('resultOriginalImage');
    const resultCamImage = document.getElementById('resultCamImage');
    const predictionClass = document.getElementById('predictionClass');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceFill = document.getElementById('confidenceFill');
    const adviceList = document.getElementById('adviceList');

    let selectedFile = null;

    // API URL
    const API_URL = 'http://127.0.0.1:8000/predict';

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid medical image format (.png, .jpg)');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        }
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI State transition
        uploadArea.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            resultOriginalImage.src = imagePreview.src;

            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Server error during inference.');
            }

            const data = await response.json();

            // Short delay to allow loading animation to process visually
            setTimeout(() => {
                showResults(data);
            }, 800);

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the backend server. Make sure the FastAPI python server is running.');
            location.reload();
        }
    });

    function showResults(data) {

        // --- NEW: Handle invalid image or backend error ---
        // If the backend rejected the image (not an X-ray, wrong format, etc.)
        // show the reason to the user and reset back to the upload screen.
        if (data.class_name === "Invalid Image" || data.class_name === "Error") {
            loadingState.classList.add('hidden');
            uploadArea.classList.remove('hidden');
            previewContainer.classList.remove('hidden');
            dropZone.classList.add('hidden');
            alert("⚠️ Invalid Image\n\n" + data.message + "\n\nPlease upload a valid bone X-ray scan.");
            return; // Stop here — do not show results
        }
        // --- END NEW BLOCK ---

        loadingState.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        const isMalignant = data.class_name.toLowerCase().includes('malignant') || data.class_name.toLowerCase().includes('tumor') || data.class_name.includes('test');

        // Update Class text and style
        predictionClass.textContent = isMalignant ? 'Detected: Malignant' : 'Detected: Benign / Normal';
        predictionClass.className = isMalignant ? 'status-badge malignant' : 'status-badge benign';

        // Apply confidence
        const confPercent = (data.confidence * 100).toFixed(1);
        confidenceText.textContent = `${confPercent}%`;

        setTimeout(() => {
            confidenceFill.style.width = `${confPercent}%`;
            confidenceFill.style.background = isMalignant ? 'var(--danger)' : 'var(--success)';
        }, 50);

        // Display Grad-CAM
        if (data.gradcam_base64) {
            resultCamImage.src = `data:image/jpeg;base64,${data.gradcam_base64}`;
        }

        // Populate Immediate Advice based on Result
        adviceList.innerHTML = '';
        const adviceItems = getAdvicePoints(isMalignant);
        adviceItems.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            adviceList.appendChild(li);
        });
    }

    function getAdvicePoints(isMalignant) {
        if (isMalignant) {
            return [
                "Urgent referral to an Orthopedic Oncologist is recommended.",
                "Schedule a follow-up biopsy to confirm histopathology.",
                "Consider a full skeletal survey or PET-CT to check for potential metastasis.",
                "Review the Grad-CAM activation map to identify the exact coordinates of the suspected lesion.",
                "Do not rely solely on AI classification; ensure multidisciplinary team (MDT) review."
            ];
        } else {
            return [
                "No immediate malignant features detected by the AI model.",
                "Correlate AI findings with the patient's clinical symptoms and history.",
                "If pain or swelling persists, consider a follow-up MRI in 3-6 months.",
                "Routinely advise the patient on bone health and activity moderation if symptomatic.",
                "Standard radiological review is still required to sign off the report."
            ];
        }
    }
});