window.addEventListener("resize", function () {
    window.resizeTo(1280, 770);
});

let settingMenuShow = true;

function toggleSettingMenu(elt) {
    elt.classList.toggle('change');
    if (settingMenuShow) {
        document.getElementById('setting-menu').style.display = '';
    } else {
        document.getElementById('setting-menu').style.display = 'none';
    }
    settingMenuShow = !settingMenuShow;
}

function pyVideo() {
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
    modelSelector.innerHTML = '';
    if (elt.value === 'classification') {
        addOptions(modelSelector, ['MobileNet V1', 'MobileNet V2', 'MobileNet V3', 'Inception V1', 'Inception V3', 'ResNet-50', 'EfficientNet']);
    } else if (elt.value === 'detection') {
        addOptions(modelSelector, ['SSD MobileNet V1', 'SSD MobileNet V2', 'SSDLite MobileDet']);
    } else { // pose-estimation
        addOptions(modelSelector, ['PoseNet MobileNet V1', 'MoveNet.SinglePose.Lightning', 'MoveNet.SinglePose.Thunder', 'PoseNet ResNet-50']);
    }
}

function onSubmitButtonClicked() {
    const inferenceType = document.getElementById('inference-type-selector').value;
    const model = document.getElementById('model-selector').value;
}