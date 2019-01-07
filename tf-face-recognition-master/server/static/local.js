//Get camera video
const constraints = {
    audio: false,
    video: {
//        width: {min: 640, ideal: 1280, max: 1920},
//        height: {min: 480, ideal: 720, max: 1080}
        width: {min: 440, ideal: 480, max: 440},

        height: {min: 380, ideal: 480, max: 480}
//        width: {min: 640, ideal: 1280, max: 1920},
//        height: {min: 480, ideal: 720, max: 1080}
    }
};

navigator.mediaDevices.getUserMedia(constraints)
    .then(stream => {
        document.getElementById("myVideo").srcObject = stream;
        console.log("Got local user video");

    })
    .catch(err => {
        console.log('navigator.getUserMedia error: ', err)
    });
