package PracticePolymorphism;

public class Hero {

    private int magicPoint;//魔法值
    private int hurt;//伤害
    private String name;//姓名

    public Hero(int magicPoint, int hurt, String name) {
        this.magicPoint = magicPoint;
        this.hurt = hurt;
        this.name = name;
    }

    public int getMagicPoint() {
        return magicPoint;
    }

    public void setMagicPoint(int magicPoint) {
        this.magicPoint = magicPoint;
    }

    public int getHurt() {
        return hurt;
    }

    public void setHurt(int hurt) {
        this.hurt = hurt;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

         //使用多态的方法
    public void attack(){
        System.out.println(this.getName() + "发动攻击，伤害为：" + this.getHurt() + ",消耗魔法值20！！！");
        this.setMagicPoint(getMagicPoint() - 20);
    }

}
