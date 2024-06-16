#include "drivermonitoringwidget.h"
#include "ui_drivermonitoringwidget.h"

DriverMonitoringWidget::DriverMonitoringWidget(DataHandler *dataHandler, ConnectionWidget *connectionWidget,SystemControl *systemControl,QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::DriverMonitoringWidget),
    dataHandler(dataHandler),
    connectionWidget(connectionWidget),
    systemControl (systemControl)
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
    //     // Display the image if available
    //     QPixmap pixmap = QPixmap::fromImage(faceImage).scaled(ui->faceLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    //     ui->faceLabel->setPixmap(pixmap);
    //     ui->faceLabel->setText(""); // Clear any previous text
    // }

            QPainter painter(&faceImage);

            // Set properties for drawing text
            painter.setPen(QPen(Qt::green));
            painter.setFont(QFont("Arial", 12));

            // Prepare the status strings using the available widget objects
            QString connectionStatus = connectionWidget->isConnected() ? "Connection Status:Connected" : "Connection Status:Disconnected";
            QString systemStatus = systemControl->systemStatus ? "System On" : "System Off";  // Assume isSystemOn() is available
            QString currentFPS = QString::number(dataHandler->lastFPS); // Assuming getCurrentFPS() returns the current FPS value
            QString activeModels1 = QString("Face: %1").arg(systemControl->getActiveFaceModel());
            QString activeModels2 = QString("Head: %1").arg(systemControl->getActiveHeadModel());
            QString activeModels3 = QString("Eye: %1").arg(systemControl->getActiveEyeModel());



            // Draw the status information on the top-left corner of the frame
            int yOffset = 15;
            int lineSpacing = 15;
            painter.drawText(10, yOffset, connectionStatus); yOffset += lineSpacing;
            painter.drawText(10, yOffset, systemStatus); yOffset += lineSpacing;
            painter.drawText(10, yOffset, "FPS: " + currentFPS); yOffset += lineSpacing;
            painter.drawText(10, yOffset, activeModels1);yOffset += lineSpacing;
            painter.drawText(10, yOffset, activeModels2);yOffset += lineSpacing;
            painter.drawText(10, yOffset, activeModels3);yOffset += lineSpacing;

            // Set the processed image to the label
            QPixmap pixmap = QPixmap::fromImage(faceImage).scaled(ui->faceLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            ui->faceLabel->setPixmap(pixmap);
            ui->faceLabel->setText(""); // Clear any previous text

        }
}
