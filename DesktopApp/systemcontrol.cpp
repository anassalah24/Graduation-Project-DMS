#include "systemcontrol.h"
#include "ui_systemcontrol.h"
#include <QButtonGroup>


SystemControl::SystemControl(DataHandler *dataHandler, ConnectionWidget *connectionWidget, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SystemControl),
    dataHandler(dataHandler),
    connectionWidget(connectionWidget)
{
    ui->setupUi(this);
    connect(ui->sendtrialbutton, &QPushButton::clicked, this, &SystemControl::onSendButtonClicked);
    // Connect the ON button signal to the slot
    connect(ui->onbutton, &QPushButton::clicked, this, &SystemControl::onSystemOnClicked);

    // Connect the OFF button signal to the slot
    connect(ui->offbutton, &QPushButton::clicked, this, &SystemControl::onSystemOffClicked);

    // Populate the combo boxes with model options
    ui->comboFaceDetection->addItems({"YoloV3 Tiny", "YoloV2", "No Face Detection"});
    ui->comboHeadPose->addItems({"AX", "AY", "AZ", "A0","eff0","eff1","eff2","eff3","No Head Pose"});
    ui->comboEyeGaze->addItems({"mobilenetv3", "squeezenet", "resnet", "mobilenet", "No Eye Gaze"});
    // Set default initial values
    ui->comboFaceDetection->setCurrentIndex(ui->comboFaceDetection->findText("YoloV2"));
    ui->comboHeadPose->setCurrentIndex(ui->comboHeadPose->findText("AY"));
    ui->comboEyeGaze->setCurrentIndex(ui->comboEyeGaze->findText("mobilenetv3"));

}

SystemControl::~SystemControl()
{
    delete ui;
}

void SystemControl::onSendButtonClicked() {
    if (!connectionWidget->isConnected()) {
        QMessageBox::warning(this, "Connection Error", "The system is disconnected. Please connect before proceeding.");
        return;
    }
    if (!systemStatus) {
        QMessageBox::warning(this, "System Status", "The system is currently off. Please turn it on before proceeding.");
        return;
    }
    // Retrieve selected options from the combo boxes
    QString faceDetectionModel = ui->comboFaceDetection->currentText();
    QString headPoseModel = ui->comboHeadPose->currentText();
    QString eyeGazeModel = ui->comboEyeGaze->currentText();

    // Construct configuration messages
    QString faceDetectionMessage = "SET_FD_MODEL:" + faceDetectionModel;
    QString headPoseMessage = "SET_HP_MODEL:" + headPoseModel;
    QString eyeGazeMessage = "SET_EG_MODEL:" + eyeGazeModel;

    // Convert messages to QByteArray
    QByteArray fdData = faceDetectionMessage.toUtf8();
    QByteArray hpData = headPoseMessage.toUtf8();
    QByteArray egData = eyeGazeMessage.toUtf8();

    // Prefix each message with its length
    int fdSize = fdData.size();
    fdData.prepend(reinterpret_cast<const char*>(&fdSize), sizeof(int));

    int hpSize = hpData.size();
    hpData.prepend(reinterpret_cast<const char*>(&hpSize), sizeof(int));

    int egSize = egData.size();
    egData.prepend(reinterpret_cast<const char*>(&egSize), sizeof(int));

    // Use DataHandler to send the messages
    dataHandler->sendData(fdData);
    dataHandler->sendData(hpData);
    dataHandler->sendData(egData);
}



void SystemControl::onSystemOnClicked() {
    if (!connectionWidget->isConnected()) {
        QMessageBox::warning(this, "Connection Error", "The system is disconnected. Please connect before proceeding.");
        return;
    }
    QString message = "TURN_ON";
    QByteArray data = message.toUtf8();

    // Prefix the message with its length
    int messageSize = data.size();
    data.prepend(reinterpret_cast<const char*>(&messageSize), sizeof(int));

    dataHandler->sendData(data);
    systemStatus = true;
}

void SystemControl::onSystemOffClicked() {
    if (!connectionWidget->isConnected()) {
        QMessageBox::warning(this, "Connection Error", "The system is disconnected. Please connect before proceeding.");
        return;
    }

    QString message = "TURN_OFF";
    QByteArray data = message.toUtf8();

    // Prefix the message with its length
    int messageSize = data.size();
    data.prepend(reinterpret_cast<const char*>(&messageSize), sizeof(int));

    dataHandler->sendData(data);
    systemStatus = false;
}


