// Smooth scrolling for navigation links
document.addEventListener('DOMContentLoaded', () => {
  // Age Estimation Webcam Functionality
  const webcamElement = document.getElementById('webcam');
  const startButton = document.getElementById('start-estimation');
  const resultPopup = document.getElementById('age-result-popup');
  const processingAnimation = document.getElementById('processing-animation');
  const ageResult = document.getElementById('age-result');
  let stream = null;
  
const buttonText = startButton.querySelector('.button-text');

// Start button click handler
startButton.addEventListener('click', function() {
  if (buttonText.textContent.trim() === 'Record') {
    // Request webcam access
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(videoStream) {
          stream = videoStream;
          webcamElement.srcObject = stream;
          webcamElement.play().catch(e => console.error("Error playing video:", e));
          
          // Change button label
          buttonText.textContent = 'Start';
          console.log("Webcam access granted, camera ready");
        })
        .catch(function(error) {
          console.error("Error accessing the webcam", error);
          alert("Could not access webcam. Please check permissions and try again.");
        });
    } else {
      alert("Your browser does not support webcam access.");
    }
  } else if (buttonText.textContent.trim() === 'Start') {
    startButton.disabled = true;
    startButton.classList.add('processing');
    startAgeEstimation();
  }
});
  
  function startAgeEstimation() {
  // Disable button and show processing state
  startButton.disabled = true;
  startButton.classList.add('processing');

  // Show popup with processing animation
  resultPopup.style.display = 'flex';
  processingAnimation.style.display = 'block';
  ageResult.style.display = 'none';

  // ---- Capture Image from Webcam ----
  const canvas = document.getElementById('capture-canvas');
  const context = canvas.getContext('2d');
  canvas.width = webcamElement.videoWidth;
  canvas.height = webcamElement.videoHeight;
  context.drawImage(webcamElement, 0, 0, canvas.width, canvas.height);

  // Convert canvas to Blob (JPEG/PNG)
  canvas.toBlob((blob) => {
    const formData = new FormData();
    formData.append('image', blob, 'capture.jpg');

    // ---- Send to Flask API ----
    fetch('/predict', {
      method: 'POST',
      body: formData
    })
      .then(response => response.json())
      .then(data => {
        // Hide animation
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
        console.error("Error sending image:", err);
        processingAnimation.style.display = 'none';
        ageResult.style.display = 'block';
        ageResult.textContent = 'Error sending image to server';
      })
      .finally(() => {
        // Re-enable button
        startButton.disabled = false;
        startButton.classList.remove('processing');
      });
  }, 'image/jpeg');
}

  
  // Close popup when ESC key is pressed
  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && resultPopup.style.display === 'flex') {
      resultPopup.style.display = 'none';
    }
  });
  
  const navLinks = document.querySelectorAll('a[href^="#"]');
  
  navLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href');
      const targetElement = document.querySelector(targetId);
      
      if (targetElement) {
        window.scrollTo({
          top: targetElement.offsetTop - 80, // Offset for header
          behavior: 'smooth'
        });
      }
    });
  });

  // Add active class to navigation links on scroll
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

  // Add responsive menu toggle for mobile
  const header = document.querySelector('header');
  
  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  });
});