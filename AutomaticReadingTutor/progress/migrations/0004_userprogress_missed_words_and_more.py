# Generated by Django 5.1.1 on 2024-09-19 10:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('progress', '0003_remove_userprogress_total_level_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprogress',
            name='missed_words',
            field=models.IntegerField(default=0, null=True),
        ),
        migrations.AlterField(
            model_name='userprogress',
            name='accuracy',
            field=models.FloatField(blank=True, default=0, null=True),
        ),
        migrations.AlterField(
            model_name='userprogress',
            name='correct_words',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='userprogress',
            name='total_words',
            field=models.IntegerField(default=0),
        ),
    ]
