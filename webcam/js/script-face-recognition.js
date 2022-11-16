let video = document.querySelector("#videoElement")

console.log('Programing...')

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
    ]).then(start)

function startVideo() {
    console.log("Start");
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia ({ video: true })
            .then(function (stream) {
                video.srcObject = stream;
            })
            .catch (function (error) {
                console.log("Something went wrong!")
            })
    } else {
        console.log("getUserMedia not supported")
    }
}


function loadLabeledImages() {
    const labels = ['Ho_Viet_Cuong', 'Ho_Ngoc_Dinh_Chien']
    return Promise.all(
        labels.map(async label => {
            const descriptions = []
            for (let i = 1; i <= 2; i++) {
                const img = await faceapi.fetchImage(`/labeled_images/${label}/${i}.jpg`)
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }
      
            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}


async function start() {
    console.log("Training data...")
    const labeledFaceDescriptors = await loadLabeledImages();
    console.log("Completed training.")

    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

    alert('Completed training.');

    startVideo();

    video.addEventListener('playing', () => {

        const canvas = faceapi.createCanvasFromMedia(video)
        document.getElementById('webcam').append(canvas)
    
        const displaySize = { width: video.width, height: video.height }
        faceapi.matchDimensions(canvas, displaySize)
    
        setInterval(async () => {
            const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors();
            const resizedDetections = faceapi.resizeResults(detections, displaySize)
    
            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    
            const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));
            
            results.forEach((result, i) => {
                const box = resizedDetections[i].detection.box;
                const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
                drawBox.draw(canvas);
            })
            // faceapi.draw.drawDetections(canvas, resizedDetections)
            // faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
            // faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
        }, 100)
    })
}
   
      