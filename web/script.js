let windowWidth = 1280;
let windowHeight = 900;

window.addEventListener("resize", function () {
    window.resizeTo(windowWidth, windowHeight);
});

let settingMenuShow = true;

function toggleSettingMenu(elt) {
    elt.classList.toggle('change');
    if (settingMenuShow) {
        document.getElementById('setting-menu').style.display = '';
        windowHeight = 1000;
        window.resizeTo(windowWidth, windowHeight);
    } else {
        document.getElementById('setting-menu').style.display = 'none';
        windowHeight = 900;
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

eel.expose(updateLog);

function updateLog(msg) {
    let elt = document.getElementById('log-console');
    const date = new Date(Date.now());
    const timedMsg = date.toUTCString() + ': ' + msg;
    elt.value += '\n' + timedMsg;
    elt.scrollTop = elt.scrollHeight;
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
            'MobileNet V1 (0.5 depth mul. 160x160)',
            'MobileNet V1 (0.25 depth mul. 128x128)',
            'MobileNet V1 (0.75 depth mul. 192x192)',
            'MobileNet V1 (1.0 depth mul. 224x224)',
            'MobileNet V2 (1.0 depth mul. 224x224)',
            'Inception V1',
            'Inception V2',
            'Inception V3',
            'Inception V4',
            'ResNet-50',
            'EfficientNet (224x224)',
            'EfficientNet (240x240)',
            'EfficientNet (300x300)']);
    } else if (elt.value === 'detection') {
        addOptions(modelSelector, [
            'SSD MobileNet V1',
            'SSD MobileNet V2',
            'SSDLite MobileDet',
            'EfficientDet-Lite0',
            'EfficientDet-Lite1',
            'EfficientDet-Lite2',
            'EfficientDet-Lite3']);
    } else if (elt.value === 'pose-estimation') {
        addOptions(modelSelector, [
            'PoseNet MobileNet V1 (353x481)',
            'PoseNet MobileNet V1 (481x641)',
            'PoseNet MobileNet V1 (721x1281)',
            'MoveNet.SinglePose.Lightning',
            'MoveNet.SinglePose.Thunder'
        ]);
    } else { // segmentation.
        addOptions(modelSelector, [
            'MobileNet V2 DeepLab V3 (0.5 depth mul)',
            'MobileNet V2 DeepLab V3 (1.0 depth mul)'
        ]);

    }
}

function setInferenceEngine() {
    const inferenceType = document.getElementById('inference-type-selector').value;
    const model = document.getElementById('model-selector').value;
    const edgetpu = document.getElementById('edgetpu-slider').checked;
    eel.set_engine(inferenceType, model, edgetpu)
}