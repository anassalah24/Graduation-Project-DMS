#include "readingswidget.h"
#include "ui_readingswidget.h"

ReadingsWidget::ReadingsWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ReadingsWidget)
{
    ui->setupUi(this);
    // Assuming labels are named and accessible as hp_1, hp_2, ..., hp_9 and eg_1, eg_2, ..., eg_9
    std::vector<QLabel*> labels = {
        ui->hp_1, ui->hp_2, ui->hp_3, ui->hp_4, ui->hp_5, ui->hp_6, ui->hp_7, ui->hp_8, ui->hp_9,
        ui->eg_1, ui->eg_2, ui->eg_3, ui->eg_4, ui->eg_5, ui->eg_6, ui->eg_7, ui->eg_8, ui->eg_9
    };

    for (QLabel* label : labels) {
        if (label != nullptr) {
            label->setText("N/A");
            label->setAlignment(Qt::AlignCenter);  // Center the text within the label


        }
    }
}

ReadingsWidget::~ReadingsWidget()
{
    delete ui;
}

void ReadingsWidget::updateLabels(const std::vector<float>& readings, const std::vector<QLabel*>& labels) {
    for (size_t i = 0; i < readings.size() && i < labels.size(); ++i) {
        if (readings[i] == -100) {
            labels[i]->setText("N/A");
            labels[i]->setAlignment(Qt::AlignCenter);  // Center the text within the label
        } else {
            float roundedValue = std::round(readings[i] * 100.0f) / 100.0f;
            labels[i]->setText(QString::number(roundedValue, 'f', 2));
            labels[i]->setAlignment(Qt::AlignCenter);  // Center the text within the label

        }
    }
}


void ReadingsWidget::displayReadings(const std::vector<std::vector<float>>& readings) {

    if (readings.empty() || readings[0].empty()) {
        // Handle the case where no readings are available
        std::vector<float> naReadings(9, -100); // Create a vector with -100 to indicate no data
        std::vector<QLabel*> labels; // Assuming you have a way to collect all relevant labels

        // You would need to actually initialize 'labels' similarly to what's done in the existing code
        // For each category of readings, repeat this initialization
        labels = {ui->hp_1, ui->hp_2, ui->hp_3, ui->hp_4, ui->hp_5, ui->hp_6, ui->hp_7, ui->hp_8, ui->hp_9};
        updateLabels(naReadings, labels);

        labels = {ui->eg_1, ui->eg_2, ui->eg_3, ui->eg_4, ui->eg_5, ui->eg_6, ui->eg_7, ui->eg_8, ui->eg_9};
        updateLabels(naReadings, labels);

        return;
    }

    if (readings.size() < 2) {
        qWarning() << "Insufficient data in readings";
        return;
    }

    const std::vector<float>& hpReadings = readings[0];
    const std::vector<float>& egReadings = readings[1];

    if (hpReadings.size() < 9 || egReadings.size() < 9) {
        qWarning() << "Insufficient readings in one of the rows";
        return;
    }

    // Assuming the labels are named hp_1, hp_2, ..., hp_9 and eg_1, eg_2, ..., eg_9 in the UI
    std::vector<QLabel*> hpLabels = {
        ui->hp_1, ui->hp_2, ui->hp_3, ui->hp_4,
        ui->hp_5, ui->hp_6, ui->hp_7, ui->hp_8, ui->hp_9
    };

    std::vector<QLabel*> egLabels = {
        ui->eg_1, ui->eg_2, ui->eg_3, ui->eg_4,
        ui->eg_5, ui->eg_6, ui->eg_7, ui->eg_8, ui->eg_9
    };

    updateLabels(hpReadings, hpLabels);
    updateLabels(egReadings, egLabels);
}
