# Generated by Django 4.2 on 2025-01-15 08:10

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('openshop_app', '0002_product_is_delete'),
    ]

    operations = [
        migrations.RenameField(
            model_name='product',
            old_name='is_delete',
            new_name='is_deleted',
        ),
    ]
