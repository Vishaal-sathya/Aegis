// Smooth scrolling for navigation links + Header behavior
document.addEventListener('DOMContentLoaded', () => {
  // ---------------- AGE ESTIMATION ----------------
  const webcamElement = document.getElementById('webcam');
  const startButton = document.getElementById('start-estimation');
  const resultPopup = document.getElementById('age-result-popup');
  const processingAnimation = document.getElementById('processing-animation');
  const ageResult = document.getElementById('age-result');
  const switchCameraButton = document.getElementById('switch-camera');
  let stream = null;
  let currentFacingMode = "user"; // default = front camera

  const buttonText = startButton?.querySelector('.button-text');

  async function startCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: currentFacingMode }
      });
      if (webcamElement) {
        webcamElement.srcObject = stream;
        await webcamElement.play();
      }
      console.log(`Camera started with facingMode=${currentFacingMode}`);
    } catch (err) {
      console.error("❌ Error accessing webcam:", err);
      alert("Could not access webcam. Please check permissions.");
    }
  }

  // Record/Start button click handler
  startButton?.addEventListener('click', async function () {
    if (buttonText.textContent.trim() === 'Record') {
      await startCamera();
      buttonText.textContent = 'Start';
    } else if (buttonText.textContent.trim() === 'Start') {
      startButton.disabled = true;
      startButton.classList.add('processing');
      startAgeEstimation();
    }
  });

  // Switch camera button handler
  switchCameraButton?.addEventListener('click', async () => {
    currentFacingMode = (currentFacingMode === "user") ? "environment" : "user";
    if (buttonText.textContent.trim() !== 'Record') {
      await startCamera();
    } else {
      console.log("⚠️ Camera not started yet. Click 'Record' first.");
    }
  });

  function startAgeEstimation() {
    startButton.disabled = true;
    startButton.classList.add('processing');
    resultPopup.style.display = 'flex';
    processingAnimation.style.display = 'block';
    ageResult.style.display = 'none';

    const canvas = document.getElementById('capture-canvas');
    const context = canvas.getContext('2d');
    if (!webcamElement.videoWidth || !webcamElement.videoHeight) {
      console.warn("⚠️ Webcam not ready for capture");
      processingAnimation.style.display = 'none';
      ageResult.style.display = 'block';
      ageResult.textContent = 'Camera not ready. Please try again.';
      startButton.disabled = false;
      startButton.classList.remove('processing');
      return;
    }

    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight;
    context.drawImage(webcamElement, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      const formData = new FormData();
      formData.append('image', blob, 'capture.jpg');

      fetch('/predict', {
        method: 'POST',
        body: formData
      })
        .then(response => response.json())
        .then(data => {
          processingAnimation.style.display = 'none';
          if (data.predicted_age) {
            ageResult.style.display = 'block';
            ageResult.textContent = data.predicted_age + ' years';
          } else {
            ageResult.style.display = 'block';
            ageResult.textContent = 'Error: ' + (data.error || 'Unknown error');
          }
        })
        .catch(err => {
          console.error("❌ Error sending image:", err);
          processingAnimation.style.display = 'none';
          ageResult.style.display = 'block';
          ageResult.textContent = 'Error sending image to server';
        })
        .finally(() => {
          startButton.disabled = false;
          startButton.classList.remove('processing');
        });
    }, 'image/jpeg');
  }

  // Close popup on ESC
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && resultPopup.style.display === 'flex') {
      resultPopup.style.display = 'none';
    }
  });

  // Smooth scrolling
  const navLinks = document.querySelectorAll('a[href^="#"]');
  navLinks.forEach(link => {
    link.addEventListener('click', function (e) {
      e.preventDefault();
      const targetId = this.getAttribute('href');
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        window.scrollTo({
          top: targetElement.offsetTop - 80,
          behavior: 'smooth'
        });
      }
    });
  });

  // Active class on scroll
  window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-link');
    let current = '';
    sections.forEach(section => {
      const sectionTop = section.offsetTop - 100;
      const sectionHeight = section.clientHeight;
      if (window.pageYOffset >= sectionTop && window.pageYOffset < sectionTop + sectionHeight) {
        current = section.getAttribute('id');
      }
    });
    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href').substring(1) === current) {
        link.classList.add('active');
      }
    });
  });

  // Header shrink on scroll
  const header = document.querySelector('header');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  });

  // ---------------- PRESENTATION ATTACK DETECTION ----------------
  const video = document.getElementById("video");
  const challengeText = document.getElementById("challenge");
  const statusText = document.getElementById("status");
  const timerText = document.getElementById("timer");
  const startBtn = document.getElementById("startBtn");

  let loop;
  let countdownLoop;
  let timeLeft = 10;
  let isPaused = false;
  let sessionActive = false;

  if (video) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        video.srcObject = stream;
        return new Promise((resolve) => {
          video.onloadedmetadata = () => {
            video.play();
            resolve();
          };
        });
      })
      .then(() => console.log("✅ PAD Webcam ready"))
      .catch((err) => {
        console.error("❌ Webcam access denied:", err);
        alert("Please allow camera access for PAD.");
      });
  }

  function captureFrame() {
    if (!video.videoWidth || !video.videoHeight) {
      console.warn("⚠️ PAD video not ready yet");
      return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg");
  }

  function startCountdown() {
    clearInterval(countdownLoop);
    timeLeft = 10;
    timerText.innerText = `⏳ Time left: ${timeLeft}s`;
    countdownLoop = setInterval(() => {
      timeLeft--;
      timerText.innerText = `⏳ Time left: ${timeLeft}s`;
      if (timeLeft <= 0) {
        clearInterval(countdownLoop);
      }
    }, 1000);
  }

  async function sendFrame() {
    if (isPaused || !sessionActive) return;

    const frame = captureFrame();
    if (!frame) return;

    try {
      let res = await fetch("/process_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: frame }),
      });

      let data = await res.json();
      console.log("PAD Response:", data);

      if (data.challenge === "done") {
        challengeText.innerText = "✅ All challenges passed!";
        statusText.innerText = data.message;
        timerText.innerText = "⏳ Finished!";
        clearInterval(loop);
        clearInterval(countdownLoop);
        sessionActive = false;
      } else if (data.challenge === "failed") {
        challengeText.innerText = "❌ Spoof Detected";
        statusText.innerText = data.message;
        timerText.innerText = "⏳ Challenge failed";
        clearInterval(loop);
        clearInterval(countdownLoop);
        sessionActive = false;
      } else {
        challengeText.innerText = "Challenge: " + data.challenge;
        statusText.innerText = "Status: " + data.message;

        if (data.passed) {
          clearInterval(countdownLoop);
          isPaused = true;

          if (data.next_challenge) {
            challengeText.innerText = `✅ Passed! Next: ${data.next_challenge} (2s...)`;
          } else {
            challengeText.innerText = `✅ Passed!`;
          }
          timerText.innerText = "⏳ Preparing next challenge...";

          setTimeout(() => {
            isPaused = false;
            startCountdown();
          }, 2000);
        }
      }
    } catch (err) {
      console.error("❌ Error sending PAD frame:", err);
      statusText.innerText = "⚠️ Connection error";
    }
  }

  startBtn?.addEventListener("click", async () => {
    try {
      await fetch("/start_session", { method: "POST" });
      sessionActive = true;

      challengeText.innerText = "Challenge: Waiting...";
      statusText.innerText = "Status: Not started";
      timerText.innerText = "⏳ Timer will start...";

      clearInterval(loop);
      clearInterval(countdownLoop);
      isPaused = false;

      loop = setInterval(sendFrame, 1500);
      startCountdown();
    } catch (err) {
      console.error("❌ Failed to start session:", err);
    }
  });
});
