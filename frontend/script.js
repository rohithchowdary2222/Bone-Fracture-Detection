const API_URL = "http://localhost:8000";
let currentUser = null;

// --- DOM Elements ---
const loginModal = document.getElementById('login-modal');
const usernameInput = document.getElementById('username');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const resultCard = document.getElementById('result-card');
const analyzeBtn = document.getElementById('analyze-btn');
const resetBtn = document.getElementById('reset-btn');
const historyGrid = document.getElementById('history-grid');

// --- Auth Functions ---
async function register() {
    const user = usernameInput.value.trim();
    const pass = document.getElementById('password').value;
    if (!user || !pass) return alert("Please enter both username and password");
    
    try {
        const response = await fetch(`${API_URL}/register`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({username: user, password: pass})
        });
        if (response.ok) {
            alert("Registration successful! You can now login.");
            toggleAuthMode();
        } else {
            const err = await response.json();
            alert(err.detail || "Registration failed");
        }
    } catch (e) {
        alert("Server error during registration");
    }
}

async function login() {
    const user = usernameInput.value.trim();
    const pass = document.getElementById('password').value;
    if (!user || !pass) return alert("Please enter both username and password");
    
    try {
        const response = await fetch(`${API_URL}/login`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({username: user, password: pass})
        });
        if (response.ok) {
            currentUser = user;
            localStorage.setItem('osteo_user', user);
            document.getElementById('user-display').innerText = user;
            document.getElementById('logout-btn').style.display = "block";
            loginModal.style.display = "none";
            loadHistory();
        } else {
            const err = await response.json();
            alert(err.detail || "Invalid credentials");
        }
    } catch (e) {
        alert("Server error during login");
    }
}

function logout() {
    currentUser = null;
    localStorage.removeItem('osteo_user');
    window.location.reload();
}

function toggleAuthMode() {
    const title = document.getElementById('auth-title');
    const btn = document.getElementById('auth-submit-btn');
    const toggle = document.getElementById('auth-toggle');
    if (title.innerText === "Welcome Back") {
        title.innerText = "Create Account";
        btn.innerText = "Register";
        btn.onclick = register;
        toggle.innerText = "Already have an account? Login";
    } else {
        title.innerText = "Welcome Back";
        btn.innerText = "Login";
        btn.onclick = login;
        toggle.innerText = "New user? Create an account";
    }
}

// Check for existing session
window.onload = () => {
    const savedUser = localStorage.getItem('osteo_user');
    if (savedUser) {
        currentUser = savedUser;
        document.getElementById('user-display').innerText = savedUser;
        document.getElementById('logout-btn').style.display = "block";
        loginModal.style.display = "none";
        loadHistory();
    }
};

// --- Upload Logic ---
dropZone.onclick = () => fileInput.click();

fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
};

dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.style.background = "rgba(255,255,255,0.08)";
};

dropZone.ondragleave = () => {
    dropZone.style.background = "transparent";
};

dropZone.ondrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
};

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.style.display = "none";
        previewContainer.style.display = "block";
        resultCard.style.display = "none";
    };
    reader.readAsDataURL(file);
}

resetBtn.onclick = () => {
    dropZone.style.display = "block";
    previewContainer.style.display = "none";
    resultCard.style.display = "none";
    fileInput.value = "";
};

// --- API Integration ---
analyzeBtn.onclick = async () => {
    if (!currentUser) return alert("Please login first");
    
    const file = fileInput.files[0];
    if (!file) return alert("Please select an image");

    analyzeBtn.innerText = "Analyzing...";
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    const modelSelect = document.getElementById('model-selector');
    const modelName = modelSelect ? modelSelect.value : 'efficientnet';

    try {
        const response = await fetch(`${API_URL}/predict?username=${currentUser}&model_name=${modelName}`, {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        showResult(data);
        loadHistory();
    } catch (error) {
        console.error("Error:", error);
        alert("Server error. Ensure the FastAPI backend is running.");
    } finally {
        analyzeBtn.innerText = "Analyze Image";
        analyzeBtn.disabled = false;
    }
};

function showResult(data) {
    resultCard.style.display = "block";
    const badge = document.getElementById('prediction-status');
    const fill = document.getElementById('confidence-fill');
    const valText = document.getElementById('confidence-value');
    const msg = document.getElementById('result-message');
    
    // Details elements
    const detailGrid = document.getElementById('detail-grid');
    const loc = document.getElementById('detail-location');
    const sev = document.getElementById('detail-severity');
    const con = document.getElementById('detail-consult');

    // Heatmap elements
    const heatmapBox = document.getElementById('heatmap-box');
    const heatmapImg = document.getElementById('heatmap-image');

    badge.innerText = data.prediction.toUpperCase();
    badge.style.background = data.prediction.toLowerCase().includes('fracture') ? '#991B1B' : '#065F46';
    
    fill.style.width = `${data.confidence}%`;
    valText.innerText = `${data.confidence}%`;
    
    if (data.prediction.toLowerCase().includes('fracture')) {
        msg.innerText = "Diagnosis complete. The analysis indicates structural discontinuity in the bone alignment.";
        if (data.details) {
            detailGrid.style.display = "grid";
            loc.innerText = data.details.location || "N/A";
            sev.innerText = data.details.severity || "N/A";
            con.innerText = data.details.specialist || "N/A";
        }
        if (data.heatmap) {
            heatmapBox.style.display = "block";
            heatmapImg.src = `data:image/jpeg;base64,${data.heatmap}`;
        } else {
            heatmapBox.style.display = "none";
        }
    } else {
        msg.innerText = "No fracture detected. The structural integrity appears normal based on the current analysis.";
        detailGrid.style.display = "none";
        heatmapBox.style.display = "none";
    }
}

let analysisChart = null;

async function loadHistory() {
    if (!currentUser) return;
    
    try {
        const response = await fetch(`${API_URL}/history/${currentUser}`);
        const data = await response.json();
        
        if (data.length > 0) {
            historyGrid.innerHTML = data.map(item => `
                <div class="history-card">
                    <div style="font-weight:bold; color: ${item.prediction.toLowerCase().includes('fracture') ? '#f87171' : '#34d399'}">
                        ${item.prediction}
                    </div>
                    <div style="font-size:0.8rem; color:#94a3b8;">Confidence: ${item.confidence}%</div>
                    <div style="font-size:0.7rem; color:#475569; margin-top:0.5rem;">${new Date(item.timestamp).toLocaleString()}</div>
                </div>
            `).join('');
            
            updateDashboard(data);
        }
    } catch (e) {
        console.log("History failed to load");
    }
}

function updateDashboard(historyData) {
    const dashboard = document.getElementById('dashboard-section');
    dashboard.style.display = "block";
    
    const fracturedCount = historyData.filter(i => i.prediction.toLowerCase().includes('fracture')).length;
    const normalCount = historyData.length - fracturedCount;
    
    const ctx = document.getElementById('analysis-chart').getContext('2d');
    
    if (analysisChart) {
        analysisChart.destroy();
    }
    
    analysisChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Fractured', 'Normal'],
            datasets: [{
                data: [fracturedCount, normalCount],
                backgroundColor: ['#ef4444', '#10b981'],
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { family: 'Inter' } }
                },
                title: {
                    display: true,
                    text: 'Cumulative Scan Statistics',
                    color: '#f8fafc',
                    font: { size: 16, family: 'Inter' }
                }
            }
        }
    });
}
async function downloadReport() {
    const badge = document.getElementById('prediction-status').innerText;
    const conf = parseFloat(document.getElementById('confidence-value').innerText);
    const loc = document.getElementById('detail-location').innerText;
    const sev = document.getElementById('detail-severity').innerText;
    const con = document.getElementById('detail-consult').innerText;
    const downloadBtn = document.getElementById('download-btn');
    
    downloadBtn.innerText = "Generating...";
    downloadBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/generate_report`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                username: currentUser,
                prediction: badge,
                confidence: conf,
                location: loc,
                severity: sev,
                specialist: con,
                image_name: "X-Ray_Scan"
            })
        });
        const data = await response.json();
        const link = document.createElement('a');
        link.href = `data:application/pdf;base64,${data.pdf_content}`;
        link.download = data.filename;
        link.click();
    } catch (e) {
        alert("Error generating report");
    } finally {
        downloadBtn.innerHTML = "📄 Download Medical Report";
        downloadBtn.disabled = false;
    }
}
