public class Tests {

    public static abstract class Car {
        abstract String drive();
        abstract String mark();
        abstract double price();
    }

    public static class Truck extends Car {

        String drive() {
            return "Truck driving...";
        }

        String mark() {
            return "Truck A";
        }

        double price() {
            return 300000;
        }
    }

    public static class Saloon extends Car {

        String drive() {
            return "Saloon driving...";
        }

        String mark() {
            return "Saloon B";
        }

        double price() {
            return 100000;
        }
    }

    public static void main(String[] args) {
        Truck t = new Truck();
        Saloon s = new Saloon();
        System.out.println("货车品牌：" + t.mark() + "，价格：" + t.price() + "，功能：" + t.drive());
        System.out.println("轿车品牌：" + s.mark() + "，价格：" + s.price() + "，功能：" + s.drive());
    }
}
