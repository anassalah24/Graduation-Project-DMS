#include "systemcontrol.h"
#include "ui_systemcontrol.h"

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
}

SystemControl::~SystemControl()
{
    delete ui;
}

void SystemControl::onSendButtonClicked() {
    QString text = ui->trialLineedit->text();
    QByteArray data = text.toUtf8();
    dataHandler->sendData(data);
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
