#include "systemcontrol.h"
#include "ui_systemcontrol.h"

SystemControl::SystemControl(DataHandler *dataHandler, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SystemControl),
    dataHandler(dataHandler)
{
    ui->setupUi(this);
    connect(ui->sendtrialbutton, &QPushButton::clicked, this, &SystemControl::onSendButtonClicked);
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
