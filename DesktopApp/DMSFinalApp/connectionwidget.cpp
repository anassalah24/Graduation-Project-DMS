#include "connectionwidget.h"
#include "ui_connectionwidget.h"
#include <QHostAddress>

ConnectionWidget::ConnectionWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ConnectionWidget),
    tcpSocket(new QTcpSocket(this))
{
    ui->setupUi(this);

    connect(ui->connectButton, &QPushButton::clicked, this, &ConnectionWidget::onConnectButtonClicked);
    connect(ui->disconnectButton, &QPushButton::clicked, this, &ConnectionWidget::onDisconnectButtonClicked);
    connect(tcpSocket, &QTcpSocket::connected, this, &ConnectionWidget::onConnected);
    connect(tcpSocket, &QTcpSocket::disconnected, this, &ConnectionWidget::onDisconnected);
}

ConnectionWidget::~ConnectionWidget() {
    delete ui;
}

void ConnectionWidget::onConnectButtonClicked() {
    QString ip = ui->ipLineEdit->text();
    quint16 port = ui->portLineEdit->text().toUShort();
    tcpSocket->connectToHost(QHostAddress(ip), port);
}

void ConnectionWidget::onDisconnectButtonClicked() {
    tcpSocket->disconnectFromHost();
}

void ConnectionWidget::onConnected() {
    ui->statusLabel->setText("Connected");
    ui->statusLabel->setStyleSheet("QLabel { background-color : green;  color : white; }");
    emit connected();
}

void ConnectionWidget::onDisconnected() {
    ui->statusLabel->setText("Disconnected");
    ui->statusLabel->setStyleSheet("QLabel { background-color : red;  color : white; }");
    emit disconnected();
}
