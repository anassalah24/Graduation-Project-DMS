#include "connectionwidget.h"
#include "ui_connectionwidget.h"
#include <QHostAddress>
#include <QMessageBox>

ConnectionWidget::ConnectionWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ConnectionWidget),
    tcpSocket1(new QTcpSocket(this)),
    tcpSocket2(new QTcpSocket(this)),
    socket1Connected(false),
    socket2Connected(false)
{
    ui->setupUi(this);
    ui->statusLabel->setText("Disconnected"); // Set initial status
    ui->statusLabel->setStyleSheet("QLabel { background-color : red; color : white; }");

    connect(ui->connectButton, &QPushButton::clicked, this, &ConnectionWidget::onConnectButtonClicked);
    connect(ui->disconnectButton, &QPushButton::clicked, this, &ConnectionWidget::onDisconnectButtonClicked);
    connect(tcpSocket1, &QTcpSocket::connected, this, &ConnectionWidget::onSocket1Connected);
    connect(tcpSocket1, &QTcpSocket::disconnected, this, &ConnectionWidget::onSocket1Disconnected);
    connect(tcpSocket2, &QTcpSocket::connected, this, &ConnectionWidget::onSocket2Connected);
    connect(tcpSocket2, &QTcpSocket::disconnected, this, &ConnectionWidget::onSocket2Disconnected);

    connect(tcpSocket1, &QTcpSocket::errorOccurred, this, &ConnectionWidget::onConnectionError);
    //connect(tcpSocket2, &QTcpSocket::errorOccurred, this, &ConnectionWidget::onConnectionError);
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
    if (socket1Connected && socket2Connected) {
        QMessageBox::warning(this, "Connection Status", "Already connected!");
        return;
    }
    QString ip = ui->ipLineEdit->text();
    quint16 port = ui->portLineEdit->text().toUShort();

    // Check if IP address or port is empty
    if (ip.isEmpty() || ui->portLineEdit->text().isEmpty()) {
        QMessageBox::warning(this, "Input Error", "Please enter both IP address and port number.");
        return;
    }

    QHostAddress address(ip);
    if (address.isNull() || address.protocol() != QAbstractSocket::IPv4Protocol) {
        QMessageBox::warning(this, "Invalid IP Address", "Please enter a valid IPv4 address. Example: 192.168.1.1");
        return;
    }

    if (port == 0 || port > 65535) {
        QMessageBox::warning(this, "Invalid Port", "Please enter a valid port number (1-65535). Example: 8080");
        return;
    }
    tcpSocket1->connectToHost(QHostAddress(ip), port);
    tcpSocket2->connectToHost(QHostAddress(ip), port + 1); // Use next port for second socket
}

void ConnectionWidget::onDisconnectButtonClicked() {
    if (!socket1Connected && !socket2Connected) {
        QMessageBox::warning(this, "Connection Status", "Already disconnected!");
        return;
    }
    if (QMessageBox::question(this, "Disconnect Confirmation", "Are you sure you want to disconnect?",
                              QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
        tcpSocket1->disconnectFromHost();
        tcpSocket2->disconnectFromHost();
    }
}

void ConnectionWidget::onConnectionError(QAbstractSocket::SocketError socketError) {
    if (socketError == QAbstractSocket::ConnectionRefusedError || socketError == QAbstractSocket::HostNotFoundError) {
        QMessageBox::critical(this, "Connection Error", "Failed to connect. Please check the IP address or port number and try again.");
    }
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

bool ConnectionWidget::isConnected() const {
    return socket1Connected && socket2Connected;
}
