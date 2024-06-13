#include "systemcontrol.h"
#include "ui_systemcontrol.h"
#include <QButtonGroup>


SystemControl::SystemControl(DataHandler *dataHandler, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SystemControl),
    dataHandler(dataHandler)
{
    ui->setupUi(this);
    connect(ui->sendtrialbutton, &QPushButton::clicked, this, &SystemControl::onSendButtonClicked);
    // Connect the ON button signal to the slot
    connect(ui->onbutton, &QPushButton::clicked, this, &SystemControl::onSystemOnClicked);

    // Connect the OFF button signal to the slot
    connect(ui->offbutton, &QPushButton::clicked, this, &SystemControl::onSystemOffClicked);


    // Creating button groups
    QButtonGroup *faceDetectionGroup = new QButtonGroup(this);
    faceDetectionGroup->addButton(ui->fd1);
    faceDetectionGroup->addButton(ui->fd2);
    faceDetectionGroup->addButton(ui->noFD);
    ui->fd1->setChecked(true); // Set first radio button as selected by default in Face Detection


    QButtonGroup *headPoseGroup = new QButtonGroup(this);
    headPoseGroup->addButton(ui->hp1);
    headPoseGroup->addButton(ui->hp2);
    headPoseGroup->addButton(ui->hp3);
    headPoseGroup->addButton(ui->noHP);
    ui->hp1->setChecked(true); // Set first radio button as selected by default in Head Pose


    QButtonGroup *eyeGazeGroup = new QButtonGroup(this);
    eyeGazeGroup->addButton(ui->eg1);
    eyeGazeGroup->addButton(ui->eg2);
    eyeGazeGroup->addButton(ui->eg3);
    eyeGazeGroup->addButton(ui->noEG);
    ui->eg1->setChecked(true); // Set first radio button as selected by default in Eye Gaze

}

SystemControl::~SystemControl()
{
    delete ui;
}

void SystemControl::onSendButtonClicked() {
    // QString text = ui->trialLineedit->text();
    // QByteArray data = text.toUtf8();
    // dataHandler->sendData(data);

    // Retrieve selected options from the UI components
    QString faceDetectionModel;
    if (ui->fd1->isChecked()) {
        faceDetectionModel = "YoloV3 Tiny";
    } else if (ui->fd2->isChecked()) {
        faceDetectionModel = "YoloV2";
    } else if (ui->noFD->isChecked()) {
        faceDetectionModel = "No Face Detection";
    }

    QString headPoseModel;
    if (ui->hp1->isChecked()) {
        headPoseModel = "AX";
    } else if (ui->hp2->isChecked()) {
        headPoseModel = "AY";
    } else if (ui->hp3->isChecked()) {
        headPoseModel = "AZ";
    } else if (ui->noHP->isChecked()) {
        headPoseModel = "No Head Pose";
    }

    QString eyeGazeModel;
    if (ui->eg1->isChecked()) {
        eyeGazeModel = "AX";
    } else if (ui->eg2->isChecked()) {
        eyeGazeModel = "AY";
    } else if (ui->eg3->isChecked()) {
        eyeGazeModel = "AZ";
    } else if (ui->noEG->isChecked()) {
        eyeGazeModel = "No Eye Gaze";
    }

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
    QString message = "TURN_ON";
    QByteArray data = message.toUtf8();

    // Prefix the message with its length
    int messageSize = data.size();
    data.prepend(reinterpret_cast<const char*>(&messageSize), sizeof(int));

    dataHandler->sendData(data);
}

void SystemControl::onSystemOffClicked() {
    QString message = "TURN_OFF";
    QByteArray data = message.toUtf8();

    // Prefix the message with its length
    int messageSize = data.size();
    data.prepend(reinterpret_cast<const char*>(&messageSize), sizeof(int));

    dataHandler->sendData(data);
}
