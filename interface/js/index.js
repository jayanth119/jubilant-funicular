var microphoneButton = document.getElementsByClassName("start-recording-button")[0];
var recordingControlButtonsContainer = document.getElementsByClassName("recording-contorl-buttons-container")[0];
var stopRecordingButton = document.getElementsByClassName("stop-recording-button")[0];
var cancelRecordingButton = document.getElementsByClassName("cancel-recording-button")[0];
var elapsedTimeTag = document.getElementsByClassName("elapsed-time")[0];
var closeBrowserNotSupportedBoxButton = document.getElementsByClassName("close-browser-not-supported-box")[0];
var overlay = document.getElementsByClassName("overlay")[0];
var audioElement = document.getElementsByClassName("audio-element")[0];
var audioElementSource = document.getElementsByClassName("audio-element")[0]
    .getElementsByTagName("source")[0];
var textIndicatorOfAudiPlaying = document.getElementsByClassName("text-indication-of-audio-playing")[0];

microphoneButton.onclick = startAudioRecording;
stopRecordingButton.onclick = stopAudioRecording;
cancelRecordingButton.onclick = cancelAudioRecording;
closeBrowserNotSupportedBoxButton.onclick = hideBrowserNotSupportedOverlay;
audioElement.onended = hideTextIndicatorOfAudioPlaying;

function handleDisplayingRecordingControlButtons() {
    microphoneButton.style.display = "none";
    recordingControlButtonsContainer.classList.remove("hide");
    handleElapsedRecordingTime();
}

function handleHidingRecordingControlButtons() {
    microphoneButton.style.display = "block";
    recordingControlButtonsContainer.classList.add("hide");
    clearInterval(elapsedTimeTimer);
}

function displayBrowserNotSupportedOverlay() {
    overlay.classList.remove("hide");
}

function hideBrowserNotSupportedOverlay() {
    overlay.classList.add("hide");
}

function createSourceForAudioElement() {
    let sourceElement = document.createElement("source");
    audioElement.appendChild(sourceElement);
    audioElementSource = sourceElement;
}

function displayTextIndicatorOfAudioPlaying() {
    textIndicatorOfAudiPlaying.classList.remove("hide");
}

function hideTextIndicatorOfAudioPlaying() {
    textIndicatorOfAudiPlaying.classList.add("hide");
}

var audioRecordStartTime;
var maximumRecordingTimeInHours = 1;
var elapsedTimeTimer;

function startAudioRecording() {
    console.log("Recording Audio...");
    let recorderAudioIsPlaying = !audioElement.paused;
    console.log("paused?", !recorderAudioIsPlaying);
    if (recorderAudioIsPlaying) {
        audioElement.pause();
        hideTextIndicatorOfAudioPlaying();
    }
    audioRecorder.start()
        .then(() => {
            audioRecordStartTime = new Date();
            handleDisplayingRecordingControlButtons();
        })
        .catch(error => {
            if (error.message.includes("mediaDevices API or getUserMedia method is not supported in this browser.")) {
                console.log("To record audio, use browsers like Chrome and Firefox.");
                displayBrowserNotSupportedOverlay();
            }
            switch (error.name) {
                case 'AbortError':
                    console.log("An AbortError has occured.");
                    break;
                case 'NotAllowedError':
                    console.log("A NotAllowedError has occured. User might have denied permission.");
                    break;
                case 'NotFoundError':
                    console.log("A NotFoundError has occured.");
                    break;
                case 'NotReadableError':
                    console.log("A NotReadableError has occured.");
                    break;
                case 'SecurityError':
                    console.log("A SecurityError has occured.");
                    break;
                case 'TypeError':
                    console.log("A TypeError has occured.");
                    break;
                case 'InvalidStateError':
                    console.log("An InvalidStateError has occured.");
                    break;
                case 'UnknownError':
                    console.log("An UnknownError has occured.");
                    break;
                default:
                    console.log("An error occured with the error name " + error.name);
            }
        });
}

function stopAudioRecording() {
    console.log("Stopping Audio Recording...");
    audioRecorder.stop()
        .then(audioAsblob => {
            playAudio(audioAsblob);
            handleHidingRecordingControlButtons();
            uploadAudio(audioAsblob); // Call the upload function
        })
        .catch(error => {
            switch (error.name) {
                case 'InvalidStateError':
                    console.log("An InvalidStateError has occured.");
                    break;
                default:
                    console.log("An error occured with the error name " + error.name);
            }
        });
}

function cancelAudioRecording() {
    console.log("Canceling audio...");
    audioRecorder.cancel();
    handleHidingRecordingControlButtons();
}

function playAudio(recorderAudioAsBlob) {
    let reader = new FileReader();
    reader.onload = (e) => {
        let base64URL = e.target.result;
        if (!audioElementSource)
            createSourceForAudioElement();
        audioElementSource.src = base64URL;
        let BlobType = recorderAudioAsBlob.type.includes(";") ?
            recorderAudioAsBlob.type.substr(0, recorderAudioAsBlob.type.indexOf(';')) : recorderAudioAsBlob.type;
        audioElementSource.type = BlobType
        audioElement.load();
        console.log("Playing audio...");
        audioElement.play();
        displayTextIndicatorOfAudioPlaying();
    };
    reader.readAsDataURL(recorderAudioAsBlob);
}

function handleElapsedRecordingTime() {
    displayElapsedTimeDuringAudioRecording("00:00");
    elapsedTimeTimer = setInterval(() => {
        let elapsedTime = computeElapsedTime(audioRecordStartTime);
        displayElapsedTimeDuringAudioRecording(elapsedTime);
    }, 1000);
}

function displayElapsedTimeDuringAudioRecording(elapsedTime) {
    elapsedTimeTag.innerHTML = elapsedTime;
    if (elapsedTimeReachedMaximumRecordingTime(elapsedTime)) {
        stopAudioRecording();
    }
}

function computeElapsedTime(startTime) {
    let endTime = new Date();
    let timeDiff = endTime - startTime;
    timeDiff = timeDiff / 1000;
    let seconds = Math.floor(timeDiff % 60);
    let secondsAsString = seconds < 10 ? `0${seconds}` : seconds;
    timeDiff = Math.floor(timeDiff / 60);
    let minutes = timeDiff % 60;
    let minutesAsString = minutes < 10 ? `0${minutes}` : minutes;
    timeDiff = Math.floor(timeDiff / 60);
    let hours = timeDiff % 60;
    let hoursAsString = hours < 10 ? `0${hours}` : hours;
    return `${hoursAsString}:${minutesAsString}:${secondsAsString}`;
}

function elapsedTimeReachedMaximumRecordingTime(elapsedTime) {
    let elapsedTimeSplitted = elapsedTime.split(":");
    let elapsedHours = parseInt(elapsedTimeSplitted[0]);
    return elapsedHours === maximumRecordingTimeInHours;
}


function uploadAudio(audioBlob) {
    var formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");
    formData.append("gender", selectedGenderSpan.textContent);

    fetch("/upload-audio/", {
        method: "POST",
        body: formData,
        headers: {
            "X-CSRFToken": getCookie("csrftoken") // Assuming you have a function to get CSRF token
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log("Success:", data);
    })
    .catch(error => {
        console.error("Error:", error);
    });
}

// Helper function to get CSRF token
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== "") {
        var cookies = document.cookie.split(";");
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + "=")) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
