from django.db import models
import uuid

class Product(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    shop = models.CharField(max_length=255)
    price = models.PositiveIntegerField()
    sku = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    location = models.CharField(max_length=255)
    discount = models.PositiveIntegerField(default=0)
    category = models.CharField(max_length=100)
    stock = models.PositiveIntegerField()
    is_available = models.BooleanField(default=True)
    is_delete = models.BooleanField(default=False)
    picture = models.URLField(max_length=500)
