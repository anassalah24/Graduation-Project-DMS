#include "connectionwidget.h"
#include "ui_connectionwidget.h"
#include <QHostAddress>

ConnectionWidget::ConnectionWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ConnectionWidget),
    tcpSocket1(new QTcpSocket(this)),
    tcpSocket2(new QTcpSocket(this)),
    socket1Connected(false),
    socket2Connected(false)
{
    ui->setupUi(this);

    connect(ui->connectButton, &QPushButton::clicked, this, &ConnectionWidget::onConnectButtonClicked);
    connect(ui->disconnectButton, &QPushButton::clicked, this, &ConnectionWidget::onDisconnectButtonClicked);
    connect(tcpSocket1, &QTcpSocket::connected, this, &ConnectionWidget::onSocket1Connected);
    connect(tcpSocket1, &QTcpSocket::disconnected, this, &ConnectionWidget::onSocket1Disconnected);
    connect(tcpSocket2, &QTcpSocket::connected, this, &ConnectionWidget::onSocket2Connected);
    connect(tcpSocket2, &QTcpSocket::disconnected, this, &ConnectionWidget::onSocket2Disconnected);
}

ConnectionWidget::~ConnectionWidget() {
    delete ui;
}

QTcpSocket* ConnectionWidget::getTcpSocket1() const {
    return tcpSocket1;
}

QTcpSocket* ConnectionWidget::getTcpSocket2() const {
    return tcpSocket2;
}

void ConnectionWidget::onConnectButtonClicked() {
    QString ip = ui->ipLineEdit->text();
    quint16 port = ui->portLineEdit->text().toUShort();
    tcpSocket1->connectToHost(QHostAddress(ip), port);
    tcpSocket2->connectToHost(QHostAddress(ip), port + 1);  // Use next port for second socket
}

void ConnectionWidget::onDisconnectButtonClicked() {
    tcpSocket1->disconnectFromHost();
    tcpSocket2->disconnectFromHost();
}

void ConnectionWidget::onSocket1Connected() {
    socket1Connected = true;
    if (socket1Connected && socket2Connected) {
        onConnected();
    }
}

void ConnectionWidget::onSocket2Connected() {
    socket2Connected = true;
    if (socket1Connected && socket2Connected) {
        onConnected();
    }
}

void ConnectionWidget::onSocket1Disconnected() {
    socket1Connected = false;
    onDisconnected();
}

void ConnectionWidget::onSocket2Disconnected() {
    socket2Connected = false;
    onDisconnected();
}

void ConnectionWidget::onConnected() {
    ui->statusLabel->setText("Connected");
    ui->statusLabel->setStyleSheet("QLabel { background-color : green; color : white; }");
    emit connected();
}

void ConnectionWidget::onDisconnected() {
    ui->statusLabel->setText("Disconnected");
    ui->statusLabel->setStyleSheet("QLabel { background-color : red; color : white; }");
    emit disconnected();
}
