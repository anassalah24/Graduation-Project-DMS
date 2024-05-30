#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    connectionWidget(new ConnectionWidget(this)),
    driverMonitoringWidget(new DriverMonitoringWidget(this)),
    systemcontrol(nullptr)

{
    ui->setupUi(this);

    // Add the widgets to the main layout
    ui->conn_layout->addWidget(connectionWidget);
    ui->dm_layout->addWidget(driverMonitoringWidget);

    // Create the DataHandler and connect it to the sockets from the ConnectionWidget
    dataHandler = new DataHandler(connectionWidget->getTcpSocket1(), connectionWidget->getTcpSocket2(), this);

    // Create the SystemControlWidget and pass the DataHandler to it
    systemcontrol = new SystemControl(dataHandler, this);
    ui->sysctrl_layout->addWidget(systemcontrol);

    // Connect the faceReceived signal to the updateFaceDisplay slot
    connect(dataHandler, &DataHandler::faceReceived, driverMonitoringWidget, &DriverMonitoringWidget::updateFaceDisplay);


}

MainWindow::~MainWindow() {
    delete ui;
}
