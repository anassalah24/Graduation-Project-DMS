#include "drivermonitoringwidget.h"
#include "ui_drivermonitoringwidget.h"

DriverMonitoringWidget::DriverMonitoringWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::DriverMonitoringWidget)
{
    ui->setupUi(this);
    updateFaceDisplay(QImage()); // Initialize with no image available

}

DriverMonitoringWidget::~DriverMonitoringWidget()
{
    delete ui;
}

void DriverMonitoringWidget::updateFaceDisplay(QImage faceImage) {
    if (faceImage.isNull()) {
        // Display a default message when no image is received
        QLabel *label = ui->faceLabel;
        label->setText("No frames available");
        label->setAlignment(Qt::AlignCenter); // Center the text within the label
        label->setStyleSheet("QLabel { background-color : black; color : white; }"); // Optional: style as needed
    } else {
        // Display the image if available
        QPixmap pixmap = QPixmap::fromImage(faceImage).scaled(ui->faceLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->faceLabel->setPixmap(pixmap);
        ui->faceLabel->setText(""); // Clear any previous text
    }
}
