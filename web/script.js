let windowWidth = 1280;
let windowHeight = 770;

window.addEventListener("resize", function () {
    window.resizeTo(windowWidth, windowHeight);
});

let settingMenuShow = true;

function toggleSettingMenu(elt) {
    elt.classList.toggle('change');
    if (settingMenuShow) {
        document.getElementById('setting-menu').style.display = '';
        windowHeight = 870;
        window.resizeTo(windowWidth, windowHeight);
    } else {
        document.getElementById('setting-menu').style.display = 'none';
        windowHeight = 770;
        window.resizeTo(windowWidth, windowHeight);
    }
    settingMenuShow = !settingMenuShow;
}

function onStart() {
    // Set Inference Engine
    setInferenceEngine()
    // Starts Video Feed.
    eel.video_feed()()
}


eel.expose(updateImageSrc);

function updateImageSrc(img) {
    let elem = document.getElementById('coral-cam-video-feed');
    elem.src = "data:image/jpeg;base64," + img;
}

function addOption(selector, optionName) {
    let option = document.createElement('option');
    option.value = optionName;
    option.textContent = optionName;
    selector.appendChild(option);
}

function addOptions(selector, options) {
    options.forEach(function (elt) {
        addOption(selector, elt);
    });
}

function inferenceSelectionChanged(elt) {
    let modelSelector = document.getElementById('model-selector');
    modelSelector.innerHTML = ''; // Clears all current model options
    if (elt.value === 'classification') {
        addOptions(modelSelector, [
            'MobileNet V1',
            'MobileNet V2',
            'Inception V1',
            'Inception V2',
            'Inception V3',
            'Inception V4',
            'ResNet-50',
            'EfficientNet (S)',
            'EfficientNet (M)',
            'EfficientNet (L)']);
    } else if (elt.value === 'detection') {
        addOptions(modelSelector, [
            'SSD MobileNet V1',
            'SSD MobileNet V2',
            'SSDLite MobileDet',
            'EfficientDet-Lite0',
            'EfficientDet-Lite1',
            'EfficientDet-Lite2',
            'EfficientDet-Lite3']);
    } else { // pose-estimation
        addOptions(modelSelector, [
            'PoseNet MobileNet V1 (S)',
            'PoseNet MobileNet V1 (M)',
            'PoseNet MobileNet V1 (L)',
            'MoveNet.SinglePose.Lightning',
            'MoveNet.SinglePose.Thunder'
        ]);
    }
}

function setInferenceEngine() {
    const inferenceType = document.getElementById('inference-type-selector').value;
    const model = document.getElementById('model-selector').value;
    const edgetpu = document.getElementById('edgetpu-slider').checked;
    eel.set_engine(inferenceType, model, edgetpu)
}