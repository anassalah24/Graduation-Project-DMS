#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    connectionWidget(new ConnectionWidget(this)),
    driverMonitoringWidget(new DriverMonitoringWidget(this))
{
    ui->setupUi(this);

    // Add the ConnectionWidget and DriverMonitoringWidget to the main layout
    ui->conn_layout->addWidget(connectionWidget);
    ui->dm_layout->addWidget(driverMonitoringWidget);

    // Create the DataHandler and connect it to the socket from the ConnectionWidget
    dataHandler = new DataHandler(connectionWidget->tcpSocket, this);

    // Connect the faceReceived signal to the updateFaceDisplay slot
    connect(dataHandler, &DataHandler::faceReceived, driverMonitoringWidget, &DriverMonitoringWidget::updateFaceDisplay);
}

MainWindow::~MainWindow() {
    delete ui;
}
