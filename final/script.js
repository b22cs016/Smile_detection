// Get references to elements
const startButton = document.getElementById('startButton');
const videoContainer = document.querySelector('.video-container');
const video = document.getElementById('video');
const smileImg = document.getElementById('smile-img');
const canvas = document.createElement('canvas'); // Create a canvas element in memory
canvas.width = 640;
canvas.height = 480;
const ctx = canvas.getContext('2d');

// Function to start the webcam
async function startWebcam() {
    console.log("Attempting to start webcam...");
    if (!video.srcObject) { // Only initialize if there's no existing stream
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            console.log("Webcam started");
        } catch (err) {
            console.error('Error accessing webcam: ', err);
        }
    } else {
        console.log("Webcam already started");
    }
}

// Function to reset the UI to initial state
function resetUI() {
    videoContainer.style.display = 'none'; // Hide the video container
    smileImg.style.display = 'none'; // Hide the smile image
    startButton.style.display = 'block'; // Show the start button again
}

// Function to download image data as a file and reset the UI
function downloadImage(dataUrl, filename) {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    alert("Your smiling face has been captured!!!\nFind it in your downloads."); // Show alert message
    resetUI();  // Call resetUI after the download
}


// Function to continuously capture frames from webcam and detect smiles
function detectSmile() {
    console.log("Detecting smiles...");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg');
    $.ajax({
        type: 'POST',
        url: '/detect_smile',
        data: { image: dataURL },
        success: function(response) {
            if (response.result === 'smile_detected') {
                // Wait for about half a second before capturing the final smiling image
                setTimeout(function() {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const delayedDataURL = canvas.toDataURL('image/jpeg');
                    downloadImage(delayedDataURL, 'smile.jpg');
                    smileImg.src = delayedDataURL;
                    smileImg.style.display = 'block';
                }, 500); // Delay in milliseconds
            } else {
                // Continue detection if no smile is detected
                setTimeout(detectSmile, 100);
            }
        }
    });
}

// Event listener for the start button
startButton.addEventListener('click', function() {
    console.log("Start button clicked");
    videoContainer.style.display = 'flex'; // Show the video container
    startButton.style.display = 'none'; // Hide the start button
    startWebcam().then(detectSmile); // Start the webcam and smile detection
});