#ifndef READINGSWIDGET_H
#define READINGSWIDGET_H

#include <QWidget>
#include <qlabel.h>

namespace Ui {
class ReadingsWidget;
}

class ReadingsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ReadingsWidget(QWidget *parent = nullptr);
    ~ReadingsWidget();

public slots:
    void displayReadings(const std::vector<std::vector<float>>& readings);

private:
    Ui::ReadingsWidget *ui;
    void updateLabels(const std::vector<float>& readings, const std::vector<QLabel*>& labels);

};

#endif // READINGSWIDGET_H
